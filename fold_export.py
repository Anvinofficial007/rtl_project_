"""
fold_export.py — Phase 3a: fold BN + quantization scales into Non-Conv (k, b),
convert them to fixed point, and export every integer parameter the hardware needs.

  k_c = gamma_c * s_a * s_w,c / (s'_a * sqrt(var_c + eps))
  b_c = (beta_c - gamma_c * mu_c / sqrt(var_c + eps)) / s'_a
  K_c = round(k_c * 2^FRAC),  B_c = round(b_c * 2^FRAC)          (Q8.16 in the paper)
  HW : y = clip((K_c * acc + B_c + 2^(FRAC-1)) >> FRAC, 0, 255)

Usage:
  python fold_export.py --ckpt ckpt/qat_best.pth
  python fold_export.py --ckpt ckpt/qat_best.pth --int-bits 9     # force wider K,B

Outputs:
  ckpt/edea_int_params.npz    all integer weights, K/B, strides, FC (input to int_model.py, Phase 4/5)
  ckpt/fold_report.json       per-layer ranges, chosen K/B width, accumulator bounds

Layouts (chosen to match the RTL/hex layout of later phases):
  stem_w  [3,3,3,32]  (kh, kw, cin, cout)   int8
  L{i}_wd [3,3,D]     (kh, kw, channel)     int8
  L{i}_wp [D,K]       (cin, cout)           int8
  K/B arrays          int64 per channel
"""
import argparse
import json
import math

import numpy as np
import torch

from qmodels import QMobileNetV1CIFAR


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/qat_best.pth")
    p.add_argument("--out", default="ckpt/edea_int_params.npz")
    p.add_argument("--report", default="ckpt/fold_report.json")
    p.add_argument("--frac-bits", type=int, default=16)
    p.add_argument("--int-bits", type=int, default=8, help="minimum integer bits (paper: 8)")
    p.add_argument("--strict", action="store_true", help="fail instead of widening K/B")
    return p.parse_args()


def step_of(lsq):
    return lsq.step.detach().double().clamp(min=1e-8)


def fold(bn, s_a, s_w, s_out):
    """Return float k, b (numpy float64, per channel)."""
    g = bn.weight.detach().double()
    beta = bn.bias.detach().double()
    mu = bn.running_mean.detach().double()
    var = bn.running_var.detach().double()
    inv = 1.0 / torch.sqrt(var + bn.eps)
    k = g * s_a * s_w * inv / s_out
    b = (beta - g * mu * inv) / s_out
    return k.numpy(), b.numpy()


def int_weight(conv):
    return conv.wq.int_repr(conv.weight.detach()).cpu().numpy().astype(np.int8)


def signed_bits_needed(v):
    """Smallest two's-complement width that holds integer v."""
    v = int(v)
    t = 1
    while not (-(1 << (t - 1)) <= v <= (1 << (t - 1)) - 1):
        t += 1
    return t


def main():
    args = get_args()
    F = args.frac_bits
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    qa = ck.get("args", {})
    model = QMobileNetV1CIFAR(per_channel=not qa.get("per_tensor", False))
    model.load_state_dict(ck["model"])
    model.eval()

    out, kb_float, report_layers = {}, {}, []

    # ---------------- stem (software, but same integer formulation) -------------- #
    s_in = step_of(model.inq).item()
    s_stem = step_of(model.stem_aq).item()
    out["s_in"] = np.float64(s_in)
    w = int_weight(model.stem_conv)                       # [32,3,3,3] (O,I,H,W)
    out["stem_w"] = w.transpose(2, 3, 1, 0).copy()        # [3,3,3,32]
    kb_float["stem"] = fold(model.stem_bn, s_in, step_of(model.stem_conv.wq), s_stem)

    # ---------------- DSC layers 0..12 ------------------------------------------ #
    s_prev = s_stem
    for i, blk in enumerate(model.features):
        s_mid, s_out = step_of(blk.aq1).item(), step_of(blk.aq2).item()
        wd = int_weight(blk.dw)                            # [D,1,3,3]
        wp = int_weight(blk.pw)                            # [K,D,1,1]
        out[f"L{i}_wd"] = wd[:, 0].transpose(1, 2, 0).copy()      # [3,3,D]
        out[f"L{i}_wp"] = wp[:, :, 0, 0].T.copy()                  # [D,K]
        out[f"L{i}_stride"] = np.int64(blk.stride)
        kb_float[f"L{i}_d"] = fold(blk.bn1, s_prev, step_of(blk.dw.wq), s_mid)
        kb_float[f"L{i}_p"] = fold(blk.bn2, s_mid, step_of(blk.pw.wq), s_out)
        out[f"L{i}_scales"] = np.array([s_prev, s_mid, s_out])     # for reference only

        # static worst-case accumulator bounds (activations <= 255)
        dw_bound = 255 * np.abs(wd.astype(np.int64)).sum(axis=(1, 2, 3)).max()
        pw_bound = 255 * np.abs(wp.astype(np.int64)).sum(axis=(1, 2, 3)).max()
        report_layers.append({"layer": i, "D": int(wd.shape[0]), "K": int(wp.shape[0]),
                              "stride": blk.stride,
                              "dwc_acc_bound": int(dw_bound),
                              "dwc_acc_bits": signed_bits_needed(dw_bound),
                              "pwc_acc_bound": int(pw_bound),
                              "pwc_acc_bits": signed_bits_needed(pw_bound)})
        s_prev = s_out

    # ---------------- FC (software) --------------------------------------------- #
    out["fc_w"] = model.fc.wq.int_repr(model.fc.weight.detach()).numpy().astype(np.int8)  # [10,1024]
    out["fc_b"] = model.fc.bias.detach().double().numpy()
    out["fc_scale"] = np.float64(s_prev * step_of(model.fc.wq).item())

    # ---------------- fixed-point K,B and width selection ------------------------ #
    needed_total = 0
    for name, (k, b) in kb_float.items():
        K = np.round(k * (1 << F)).astype(np.int64)
        B = np.round(b * (1 << F)).astype(np.int64)
        kb_float[name] = (k, b, K, B)
        for v in (K.max(), K.min(), B.max(), B.min()):
            needed_total = max(needed_total, signed_bits_needed(v))
    needed_int = needed_total - F
    if needed_int > args.int_bits and args.strict:
        raise SystemExit(f"K/B need {needed_int} integer bits > {args.int_bits} (--strict)")
    int_bits = max(args.int_bits, needed_int)
    width = int_bits + F

    coef_rows = []
    for name, (k, b, K, B) in kb_float.items():
        prefix = name.replace("_d", "").replace("_p", "")
        suffix = {"stem": ("stem_K", "stem_B")}.get(name, None)
        if suffix is None:
            side = "d" if name.endswith("_d") else "p"
            suffix = (f"{prefix}_K{side}", f"{prefix}_B{side}")
        out[suffix[0]], out[suffix[1]] = K, B
        nz = np.abs(k) > 0
        rel_err = np.abs(K[nz] / (1 << F) - k[nz]) / np.abs(k[nz])
        coef_rows.append({
            "name": name, "k_min": float(k.min()), "k_max": float(k.max()),
            "b_min": float(b.min()), "b_max": float(b.max()),
            "min_abs_K": int(np.abs(K[K != 0]).min()) if (K != 0).any() else 0,
            "k_max_rel_err": float(rel_err.max()) if rel_err.size else 0.0,
            "neg_k_channels": int((k < 0).sum()),
            "overflow_q8_16": int(((K >= 1 << 23) | (K < -(1 << 23)) |
                                   (B >= 1 << 23) | (B < -(1 << 23))).sum()),
        })

    out["frac_bits"] = np.int64(F)
    out["kb_int_bits"] = np.int64(int_bits)
    out["kb_width"] = np.int64(width)
    np.savez(args.out, **out)

    # ---------------- report ----------------------------------------------------- #
    print(f"{'coef set':<8} {'k min':>10} {'k max':>10} {'b min':>9} {'b max':>9} "
          f"{'min|K|':>7} {'k relerr':>9} {'k<0':>4} {'ovf Q8.16':>9}")
    for r in coef_rows:
        print(f"{r['name']:<8} {r['k_min']:10.5f} {r['k_max']:10.5f} {r['b_min']:9.2f} "
              f"{r['b_max']:9.2f} {r['min_abs_K']:7d} {r['k_max_rel_err']:9.2e} "
              f"{r['neg_k_channels']:4d} {r['overflow_q8_16']:9d}")
    print(f"\nK/B need {needed_int} integer bits -> using Q{int_bits}.{F} ({width}-bit)"
          + ("   [matches paper Q8.16]" if width == 24 else "   [WIDER than paper: use this width in RTL]"))

    print(f"\n{'L':>2} {'D':>5} {'K':>5} {'DWC acc bound':>14} {'bits':>5} {'PWC acc bound':>14} {'bits':>5}")
    for r in report_layers:
        print(f"{r['layer']:>2} {r['D']:>5} {r['K']:>5} {r['dwc_acc_bound']:>14,} "
              f"{r['dwc_acc_bits']:>5} {r['pwc_acc_bound']:>14,} {r['pwc_acc_bits']:>5}")
    assert all(r["dwc_acc_bits"] <= 24 for r in report_layers), "DWC accumulator > 24 bits!"
    assert all(r["pwc_acc_bits"] <= 32 for r in report_layers), "PWC accumulator > 32 bits!"
    print("Accumulator widths: DWC <= 24 bit, PWC <= 32 bit  -> OK (guaranteed for any input)")

    with open(args.report, "w") as f:
        json.dump({"frac_bits": F, "kb_int_bits": int_bits, "kb_width": width,
                   "coefficients": coef_rows, "layers": report_layers,
                   "qat_acc": ck.get("summary", {}).get("qat_acc")}, f, indent=2)
    print(f"\nsaved {args.out} and {args.report}")


if __name__ == "__main__":
    main()
