"""
int_model.py — Phase 3b: bit-exact INTEGER golden model of the network.

Everything between the int8 input image and the final 2x2x1024 uint8 feature map is
computed with integers only, using exactly the arithmetic the RTL will implement:

  DWC  : acc24 = sum_{3x3} u8 * s8            (zero padding = code 0)
  NC   : y = clip((acc * K + B + 2^(F-1)) >> F, 0, 255)   (>> = arithmetic shift)
  PWC  : acc32 = sum_D u8 * s8
  NC   : same formula with the PWC-side K,B
Stem conv (software) uses the same formulation. Avg-pool + FC are done in float (software).

Speed trick: matrix products are done in float64 BLAS and converted back. This is EXACT
because every partial sum is < 2^53 (|acc| < 2^31 here).

Usage:
  python int_model.py --selftest                       # no checkpoint needed
  python int_model.py --params ckpt/edea_int_params.npz --ckpt ckpt/qat_best.pth
  python int_model.py ... --n-images 1000 --dump 16    # also save golden samples

Outputs:
  ckpt/int_model_report.json    accuracies, agreement, per-boundary code mismatch
  ckpt/golden_samples.npz       (with --dump N) int8 inputs + all layer codes for N images
"""
import argparse
import json

import numpy as np

ACC_DWC_BITS, ACC_PWC_BITS = 24, 32


# --------------------------------------------------------------------------- #
# Integer primitives (these are the reference for the RTL blocks)
# --------------------------------------------------------------------------- #
def nonconv(acc, K, B, frac=16):
    """Fused dequant+BN+ReLU+requant. acc: int64 [..., C]; K,B: int64 [C]."""
    acc = acc.astype(np.int64)
    y = (acc * K + B + (1 << (frac - 1))) >> frac          # numpy >> on int64 = arithmetic
    return np.clip(y, 0, 255).astype(np.uint8)


def _exact_matmul(a, w):
    """Integer matmul via float64 (exact while |result| < 2^53)."""
    return np.rint(np.matmul(a.astype(np.float64), w.astype(np.float64))).astype(np.int64)


def dwc_int(x, w, stride):
    """x: [B,H,W,D] uint8, w: [3,3,D] int8 -> acc int64 [B,H/s,W/s,D]."""
    Bn, H, W, D = x.shape
    Ho, Wo = H // stride, W // stride
    xp = np.pad(x.astype(np.int64), ((0, 0), (1, 1), (1, 1), (0, 0)))
    acc = np.zeros((Bn, Ho, Wo, D), np.int64)
    for i in range(3):
        for j in range(3):
            acc += xp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride, :] * w[i, j, :].astype(np.int64)
    return acc


def pwc_int(a, w):
    """a: [B,H,W,D] uint8, w: [D,K] int8 -> acc int64 [B,H,W,K]."""
    return _exact_matmul(a, w)


def conv3x3_int(x, w):
    """Standard 3x3 conv, stride 1, pad 1. x: [B,H,W,Cin] int, w: [3,3,Cin,Cout]."""
    Bn, H, W, _ = x.shape
    xp = np.pad(x.astype(np.int64), ((0, 0), (1, 1), (1, 1), (0, 0)))
    acc = np.zeros((Bn, H, W, w.shape[3]), np.int64)
    for i in range(3):
        for j in range(3):
            acc += _exact_matmul(xp[:, i:i + H, j:j + W, :], w[i, j])
    return acc


def check_range(acc, bits, what):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    mn, mx = int(acc.min()), int(acc.max())
    assert lo <= mn and mx <= hi, f"{what}: accumulator [{mn},{mx}] exceeds {bits}-bit"
    return max(abs(mn), abs(mx))


# --------------------------------------------------------------------------- #
# Integer network
# --------------------------------------------------------------------------- #
class IntMobileNet:
    def __init__(self, npz_path):
        p = np.load(npz_path)
        self.p = {k: p[k] for k in p.files}
        self.F = int(self.p["frac_bits"])
        self.n_layers = sum(1 for k in self.p if k.endswith("_wd"))
        self.acc_peak = {}

    def dsc_layer(self, x, i, keep=None):
        p = self.p
        acc_d = dwc_int(x, p[f"L{i}_wd"], int(p[f"L{i}_stride"]))
        self.acc_peak[f"L{i}_dwc"] = max(self.acc_peak.get(f"L{i}_dwc", 0),
                                         check_range(acc_d, ACC_DWC_BITS, f"L{i} DWC"))
        a = nonconv(acc_d, p[f"L{i}_Kd"], p[f"L{i}_Bd"], self.F)
        acc_p = pwc_int(a, p[f"L{i}_wp"])
        self.acc_peak[f"L{i}_pwc"] = max(self.acc_peak.get(f"L{i}_pwc", 0),
                                         check_range(acc_p, ACC_PWC_BITS, f"L{i} PWC"))
        y = nonconv(acc_p, p[f"L{i}_Kp"], p[f"L{i}_Bp"], self.F)
        if keep is not None:
            keep[f"features.{i}.aq1"] = a
            keep[f"features.{i}.aq2"] = y
        return y

    def stem(self, x_int8):
        acc = conv3x3_int(x_int8, self.p["stem_w"])
        return nonconv(acc, self.p["stem_K"], self.p["stem_B"], self.F)

    def head(self, feat_u8):
        pooled = feat_u8.astype(np.int64).sum(axis=(1, 2))                   # [B,1024]
        hw = feat_u8.shape[1] * feat_u8.shape[2]
        return (self.p["fc_scale"] / hw) * _exact_matmul(pooled, self.p["fc_w"].T) + self.p["fc_b"]

    def forward(self, x_int8, keep=None):
        """x_int8: [B,32,32,3] int8 (NHWC). Returns logits; fills `keep` with all codes."""
        x = self.stem(x_int8)
        if keep is not None:
            keep["stem_aq"] = x
        for i in range(self.n_layers):
            x = self.dsc_layer(x, i, keep)
        return self.head(x)


# --------------------------------------------------------------------------- #
# Self-test of the primitives against naive loops (no checkpoint needed)
# --------------------------------------------------------------------------- #
def selftest(seed=0):
    rng = np.random.default_rng(seed)
    # nonconv corner cases
    F = 16
    acc = np.array([-100000, -1, 0, 1, 1000, 123456, 10**8], np.int64)
    K = np.full(7, 3 << 15, np.int64)          # k = 1.5
    B = np.full(7, 1 << 15, np.int64)          # b = 0.5
    ref = [min(max(int(np.floor(a * 1.5 + 0.5 + 0.5)), 0), 255) for a in acc]
    assert nonconv(acc, K, B, F).tolist() == ref, "nonconv rounding/clip mismatch"
    # negative-k, exact .5 cases: k = -0.5, b = 0
    acc2 = np.array([-3, -1, 1, 3], np.int64)
    y2 = nonconv(acc2, np.full(4, -(1 << 15), np.int64), np.zeros(4, np.int64), F)
    assert y2.tolist() == [2, 1, 0, 0], y2      # -0.5*-3=1.5->2, 0.5->1 (round half up)

    for stride in (1, 2):
        for H in (2, 4, 8):
            if H // stride == 0:
                continue
            x = rng.integers(0, 256, (2, H, H, 8), dtype=np.int64).astype(np.uint8)
            w = rng.integers(-127, 128, (3, 3, 8)).astype(np.int8)
            got = dwc_int(x, w, stride)
            Ho = H // stride
            ref = np.zeros_like(got)
            for b in range(2):
                for n in range(Ho):
                    for m in range(Ho):
                        for d in range(8):
                            s = 0
                            for i in range(3):
                                for j in range(3):
                                    r, c = stride * n + i - 1, stride * m + j - 1
                                    if 0 <= r < H and 0 <= c < H:
                                        s += int(x[b, r, c, d]) * int(w[i, j, d])
                            ref[b, n, m, d] = s
            assert np.array_equal(got, ref), f"dwc mismatch s={stride} H={H}"

    a = rng.integers(0, 256, (2, 2, 2, 1024)).astype(np.uint8)
    w = rng.integers(-127, 128, (1024, 64)).astype(np.int8)
    ref = np.einsum("bhwd,dk->bhwk", a.astype(np.int64), w.astype(np.int64))
    assert np.array_equal(pwc_int(a, w), ref), "pwc mismatch"
    worst = np.full((1, 1, 1, 1024), 255, np.uint8)
    assert pwc_int(worst, np.full((1024, 1), -128, np.int8))[0, 0, 0, 0] == -255 * 128 * 1024

    x = rng.integers(-127, 128, (2, 6, 6, 3)).astype(np.int8)
    w = rng.integers(-127, 128, (3, 3, 3, 5)).astype(np.int8)
    got = conv3x3_int(x, w)
    xp = np.pad(x.astype(np.int64), ((0, 0), (1, 1), (1, 1), (0, 0)))
    ref = sum(np.einsum("bhwc,co->bhwo", xp[:, i:i + 6, j:j + 6], w[i, j].astype(np.int64))
              for i in range(3) for j in range(3))
    assert np.array_equal(got, ref), "stem conv mismatch"
    print("int_model self-test: PASS (nonconv rounding/clip, DWC s=1/2 incl. padding, PWC, stem)")


# --------------------------------------------------------------------------- #
# Compare integer model with the QAT (fake-quant) PyTorch model on CIFAR-10
# --------------------------------------------------------------------------- #
def compare(args):
    import torch
    import torchvision
    import torchvision.transforms as T
    from qmodels import QMobileNetV1CIFAR, lsq_modules
    from train_float import CIFAR_MEAN, CIFAR_STD

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    qmodel = QMobileNetV1CIFAR(per_channel=not ck.get("args", {}).get("per_tensor", False)).to(dev)
    qmodel.load_state_dict(ck["model"])
    qmodel.eval()
    imodel = IntMobileNet(args.params)

    # capture fake-quant codes at every hardware boundary
    tq, hooks = {}, []
    for name, m in lsq_modules(qmodel, "act"):
        if not m.signed:
            hooks.append(m.register_forward_hook(
                lambda mod, _i, out, n=name: tq.__setitem__(
                    n, mod.int_repr(out).permute(0, 2, 3, 1).cpu().numpy())))

    ds = torchvision.datasets.CIFAR10(args.data, train=False, download=True,
                                      transform=T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False)

    n = c_int = c_q = agree = 0
    mism = {}
    dump = {"x_int8": [], "label": []}
    for x, y in loader:
        if n >= args.n_images:
            break
        x, y = x[: args.n_images - n], y[: args.n_images - n]
        with torch.no_grad():
            x_int8 = qmodel.inq.int_repr(x.to(dev)).permute(0, 2, 3, 1).cpu().numpy().astype(np.int8)
            pred_q = qmodel(x.to(dev)).argmax(1).cpu().numpy()
        keep = {}
        pred_i = imodel.forward(x_int8, keep).argmax(1)
        yl = y.numpy()
        c_int += (pred_i == yl).sum(); c_q += (pred_q == yl).sum(); agree += (pred_i == pred_q).sum()
        for k, v in keep.items():
            d = np.abs(v.astype(np.int32) - tq[k].astype(np.int32))
            s = mism.setdefault(k, [0, 0, 0])
            s[0] += d.size; s[1] += int((d > 0).sum()); s[2] = max(s[2], int(d.max()))
        if len(dump["label"]) < args.dump:
            take = min(args.dump - len(dump["label"]), len(yl))
            dump["x_int8"].append(x_int8[:take]); dump["label"].extend(yl[:take].tolist())
            for k, v in keep.items():
                dump.setdefault(k, []).append(v[:take])
        n += len(yl)
        print(f"\r{n} images | INT acc {100 * c_int / n:.2f}% | QAT acc {100 * c_q / n:.2f}% "
              f"| agreement {100 * agree / n:.2f}%", end="")
    print()
    for h in hooks:
        h.remove()

    print(f"\n{'boundary':<20} {'code mismatch %':>16} {'max |diff|':>11}")
    order = ["stem_aq"] + [f"features.{i}.{q}" for i in range(imodel.n_layers) for q in ("aq1", "aq2")]
    for k in order:
        s = mism[k]
        print(f"{k:<20} {100 * s[1] / s[0]:16.4f} {s[2]:11d}")
    print("\npeak |accumulator| per layer:")
    for k, v in imodel.acc_peak.items():
        print(f"  {k:<8} {v:>12,}  ({int(v).bit_length() + 1} bits)")

    report = {"n_images": n, "int_acc": 100 * c_int / n, "qat_acc": 100 * c_q / n,
              "agreement": 100 * agree / n,
              "code_mismatch_pct": {k: 100 * mism[k][1] / mism[k][0] for k in order},
              "code_max_diff": {k: mism[k][2] for k in order},
              "acc_peak": {k: int(v) for k, v in imodel.acc_peak.items()}}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    if args.dump:
        np.savez(args.golden, **{k: (np.concatenate(v) if k != "label" else np.array(v))
                                 for k, v in dump.items()})
        print(f"saved {args.dump} golden samples -> {args.golden}")
    print(f"saved {args.report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--params", default="ckpt/edea_int_params.npz")
    ap.add_argument("--ckpt", default="ckpt/qat_best.pth")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--n-images", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--dump", type=int, default=16, help="save N golden samples (0 = off)")
    ap.add_argument("--golden", default="ckpt/golden_samples.npz")
    ap.add_argument("--report", default="ckpt/int_model_report.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        selftest()
        compare(a)
