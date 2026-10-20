"""
qmodels.py — Phase 2: LSQ fake-quantization + quantized MobileNetV1-CIFAR.

Quantizers are placed exactly at the hardware's integer boundaries:

  image ──inq(s8)──► stem_conv(wq s8) ─ BN ─ ReLU ─ stem_aq(u8) ═══► DSC layer 0 input
  DSC block i (= one accelerator pass):
     x(u8) ─► dw(wq s8, per-ch) ─ bn1 ─ ReLU ─ aq1(u8)      ← DWC Non-Conv output / PWC input
           ─► pw(wq s8, per-ch) ─ bn2 ─ ReLU ─ aq2(u8)      ← block output / next DWC input
  avgpool ─► fc(wq s8)  (software)

Phase 3 folds, per channel c of each block:
  DWC side : s_a = prev aq2 (or stem_aq) step, s_w = dw.wq.step[c], s'_a = aq1 step, BN = bn1
  PWC side : s_a = aq1 step,                 s_w = pw.wq.step[c], s'_a = aq2 step, BN = bn2
Module names are identical to models.py, so float checkpoints load directly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DSC_CFG


# --------------------------------------------------------------------------- #
# LSQ (Esser et al., "Learned Step Size Quantization", 2019)
# --------------------------------------------------------------------------- #
def grad_scale(x, g):
    """Forward: x. Backward: gradient multiplied by g."""
    return (x - x * g).detach() + x * g


def round_pass(x):
    """Forward: round(x). Backward: straight-through (identity)."""
    return (x.round() - x).detach() + x


class LSQ(nn.Module):
    """
    Fake quantizer  x_hat = clip(round(x / s), Qn, Qp) * s  with learnable s.

    kind     : "weight" or "act" (only used for reporting / calibration)
    signed   : weights and the input image are signed; post-ReLU activations unsigned
    channels : 1 = per-tensor, else per-channel along ch_axis (weights: out channel)
    """

    def __init__(self, bits=8, signed=True, channels=1, ch_axis=0,
                 kind="act", symmetric=True):
        super().__init__()
        if signed:
            self.Qn = -(2 ** (bits - 1)) + (1 if symmetric else 0)   # -127 (symmetric)
            self.Qp = 2 ** (bits - 1) - 1                            # 127
        else:
            self.Qn, self.Qp = 0, 2 ** bits - 1                      # 0..255
        self.bits, self.signed, self.kind = bits, signed, kind
        self.channels, self.ch_axis = channels, ch_axis
        self.step = nn.Parameter(torch.ones(channels))
        self.register_buffer("initialized", torch.zeros(1))
        # calibration state (activations)
        self.calibrating = False
        self.calib_mode = "percentile"
        self._calib_vals = []

    # ---- helpers ---------------------------------------------------------- #
    def _shape(self, s, x):
        if self.channels == 1:
            return s
        shape = [1] * x.dim()
        shape[self.ch_axis] = -1
        return s.view(shape)

    def _lsq_init(self, x):
        """LSQ paper init: 2 * mean(|x|) / sqrt(Qp) (per channel if needed)."""
        if self.channels == 1:
            return 2 * x.abs().mean() / math.sqrt(self.Qp)
        dims = [d for d in range(x.dim()) if d != self.ch_axis]
        return 2 * x.abs().mean(dim=dims) / math.sqrt(self.Qp)

    def _percentile_init(self, x, q=0.9999):
        v = x.detach().abs().flatten().float()
        if v.numel() > 2_000_000:
            v = v[torch.randint(0, v.numel(), (2_000_000,), device=v.device)]
        return torch.quantile(v, q) / self.Qp

    @torch.no_grad()
    def init_from(self, x):
        val = self._lsq_init(x.detach().float())
        self.step.data.copy_(val.clamp(min=1e-8).reshape(self.step.shape))
        self.initialized.fill_(1)

    @torch.no_grad()
    def finish_calibration(self):
        if self._calib_vals:
            v = torch.stack(self._calib_vals).mean()
            self.step.data.fill_(max(v.item(), 1e-8))
            self.initialized.fill_(1)
        self._calib_vals, self.calibrating = [], False

    # ---- forward ---------------------------------------------------------- #
    def forward(self, x):
        if self.calibrating:                       # pass-through, collect statistics
            with torch.no_grad():
                if self.calib_mode == "percentile":
                    self._calib_vals.append(self._percentile_init(x))
                else:
                    self._calib_vals.append(self._lsq_init(x.float()))
            return x
        if self.initialized.item() == 0:           # fallback: init on first use
            self.init_from(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            n_per = x32.numel() / self.channels
            g = 1.0 / math.sqrt(n_per * self.Qp)
            s = grad_scale(self.step.clamp(min=1e-8), g)
            s = self._shape(s, x32)
            q = round_pass(torch.clamp(x32 / s, self.Qn, self.Qp))
            return (q * s).to(x.dtype)

    @torch.no_grad()
    def int_repr(self, x):
        """Integer codes as the hardware sees them (used in Phase 3/5)."""
        s = self._shape(self.step.clamp(min=1e-8), x)
        return torch.clamp(torch.round(x / s), self.Qn, self.Qp).to(torch.int32)

    def extra_repr(self):
        return (f"kind={self.kind}, bits={self.bits}, signed={self.signed}, "
                f"range=[{self.Qn},{self.Qp}], channels={self.channels}")


# --------------------------------------------------------------------------- #
# Quantized layers
# --------------------------------------------------------------------------- #
class QConv2d(nn.Conv2d):
    def __init__(self, *args, w_bits=8, per_channel=True, **kwargs):
        super().__init__(*args, **kwargs)
        ch = self.out_channels if per_channel else 1
        self.wq = LSQ(w_bits, signed=True, channels=ch, ch_axis=0, kind="weight")

    def forward(self, x):
        return self._conv_forward(x, self.wq(self.weight), self.bias)


class QLinear(nn.Linear):
    def __init__(self, *args, w_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.wq = LSQ(w_bits, signed=True, channels=1, kind="weight")

    def forward(self, x):
        return F.linear(x, self.wq(self.weight), self.bias)


class QDSCBlock(nn.Module):
    def __init__(self, cin, cout, stride, per_channel=True):
        super().__init__()
        self.stride = stride
        self.dw = QConv2d(cin, cin, 3, stride, 1, groups=cin, bias=False, per_channel=per_channel)
        self.bn1 = nn.BatchNorm2d(cin)
        self.aq1 = LSQ(8, signed=False, kind="act")       # DWC Non-Conv output (PWC input)
        self.pw = QConv2d(cin, cout, 1, 1, 0, bias=False, per_channel=per_channel)
        self.bn2 = nn.BatchNorm2d(cout)
        self.aq2 = LSQ(8, signed=False, kind="act")       # block output (next layer input)

    def forward(self, x):
        x = self.aq1(F.relu(self.bn1(self.dw(x))))
        x = self.aq2(F.relu(self.bn2(self.pw(x))))
        return x


class QMobileNetV1CIFAR(nn.Module):
    def __init__(self, num_classes=10, stem_ch=32, per_channel=True):
        super().__init__()
        self.inq = LSQ(8, signed=True, kind="act")           # normalized image -> int8
        self.stem_conv = QConv2d(3, stem_ch, 3, 1, 1, bias=False, per_channel=per_channel)
        self.stem_bn = nn.BatchNorm2d(stem_ch)
        self.stem_aq = LSQ(8, signed=False, kind="act")      # DSC layer-0 input (uint8)

        layers, cin = [], stem_ch
        for cout, s in DSC_CFG:
            layers.append(QDSCBlock(cin, cout, s, per_channel))
            cin = cout
        self.features = nn.Sequential(*layers)
        self.fc = QLinear(cin, num_classes)

    def forward(self, x):
        x = self.inq(x)
        x = self.stem_aq(F.relu(self.stem_bn(self.stem_conv(x))))
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_float_weights(qmodel, float_state):
    """Copy float weights/BN into the quantized model; only quantizer params may be missing."""
    missing, unexpected = qmodel.load_state_dict(float_state, strict=False)
    bad = [k for k in missing if not (k.endswith(".step") or k.endswith(".initialized"))]
    assert not bad and not unexpected, f"state mismatch: missing={bad} unexpected={unexpected}"


def lsq_modules(model, kind=None):
    for name, m in model.named_modules():
        if isinstance(m, LSQ) and (kind is None or m.kind == kind):
            yield name, m


@torch.no_grad()
def init_weight_steps(model):
    """Weight steps initialised deterministically from the float weights."""
    for name, m in model.named_modules():
        if isinstance(m, (QConv2d, QLinear)):
            m.wq.init_from(m.weight)


@torch.no_grad()
def calibrate_activations(model, loader, device, n_batches=20, mode="percentile"):
    """Initialise activation steps from data (BN in eval mode, running stats)."""
    model.eval()
    for _, m in lsq_modules(model, "act"):
        m.calibrating, m.calib_mode, m._calib_vals = True, mode, []
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x.to(device))
    for _, m in lsq_modules(model, "act"):
        m.finish_calibration()


def set_bn_eval(model):
    """Freeze BN running statistics (call after model.train())."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def quant_report(model):
    print(f"{'quantizer':<24} {'kind':<6} {'range':<11} {'step (min / max)':>26}")
    for name, m in lsq_modules(model):
        s = m.step.detach()
        print(f"{name:<24} {m.kind:<6} [{m.Qn},{m.Qp}]".ljust(43) +
              f"{s.min().item():12.6f} / {s.max().item():12.6f}")


@torch.no_grad()
def activation_stats(model, loader, device, n_batches=20):
    """
    Zero percentage and saturation (code == 255) per activation quantizer.
    Zero % is the paper's Fig. 11 metric (DWC input = previous aq2, PWC input = aq1).
    """
    model.eval()
    acc, hooks = {}, []

    def make_hook(name, mod):
        def hook(_, __, out):
            codes = mod.int_repr(out)
            a = acc.setdefault(name, [0, 0, 0])
            a[0] += codes.numel()
            a[1] += (codes == 0).sum().item()
            a[2] += (codes == mod.Qp).sum().item()
        return hook

    for name, m in lsq_modules(model, "act"):
        if not m.signed:
            hooks.append(m.register_forward_hook(make_hook(name, m)))
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x.to(device))
    for h in hooks:
        h.remove()
    return {k: {"zero_pct": 100 * v[1] / v[0], "sat_pct": 100 * v[2] / v[0]} for k, v in acc.items()}
