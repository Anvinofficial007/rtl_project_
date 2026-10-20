"""
dataflow_model.py — Phase 4: tile-level / cycle-level model of the EDEA accelerator.

It executes each DSC layer with EXACTLY the loop nest, buffers and engine shapes the
RTL will have, checks every result bit-exactly against the integer golden model,
and reports cycles, throughput, PE utilization and memory traffic (paper Eq.1/2,
Table II, Fig. 3, Fig. 10, Fig. 13).

Hardware schedule (one layer):
  for tile (tr, tc)            output tile T x T, T = min(8, N)       n_tiles = (N/T)^2
    for dg in D/Td             D-group of Td = 8 channels
      [load ifmap tile T_in x T_in x 8 (T_in = s(T-1)+3), DW weights, PW rows, K/B]
      pipeline fill: FILL cycles (paper: 9)
      for pos (pr, pc)         2x2 output block inside the tile          (T/2)^2
        DWC engine: 2x2x8 acc (288 MAC) -> Non-Conv -> intermediate buffer
        for kg in K/Tk         Tk = 16 kernels
          PWC engine: 2x2x16 (512 MAC); psum = (dg==0 ? new : psum + new)
          if dg is last: output requant -> ofmap
  Lat_tile = FILL + (T/2)^2 * K/16    Lat_total = Lat_tile * n_tiles * D/8

Usage:
  python dataflow_model.py --selftest                     # random weights, all 13 layer shapes
  python dataflow_model.py                                # bit-exact vs ckpt/golden_samples.npz + report
  python dataflow_model.py --plots --trace-layer 12       # also PNG figures + event trace CSV
"""
import argparse
import csv
import json
import os

import numpy as np

from int_model import nonconv, dwc_int, pwc_int

TD, TK, TN, TM = 8, 16, 2, 2
TILE_MAX = 8
DWC_MACS, PWC_MACS = TD * 9 * TN * TM, TD * TK * TN * TM      # 288, 512
PAPER_GOPS = [1024.0] * 5 + [973.5] * 6 + [905.6] * 2          # Fig. 13
DSC_CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
           (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)]


# --------------------------------------------------------------------------- #
# Engines (shapes identical to the RTL)
# --------------------------------------------------------------------------- #
def dwc_engine(win5, wd8, s):
    """
    win5: [5,5,8] uint8 window (only 4x4 used when s=1), wd8: [3,3,8] int8.
    8 PEs (channels) x 4 columns (output pixels) x 9 multipliers = 288 MACs.
    Output pixel (r,c) uses win5[s*r+i, s*c+j] for tap (i,j).
    """
    out = np.zeros((2, 2, TD), np.int64)
    for i in range(3):
        for j in range(3):
            out += win5[i:i + s + 1:s, j:j + s + 1:s, :].astype(np.int64) * wd8[i, j].astype(np.int64)
    return out


def pwc_engine(a, w8x16):
    """
    a: [2,2,8] uint8 (intermediate buffer entry), w8x16: [8,16] int8 (PW weight row kg).
    128 PEs (16 kernels x 8 channels) x 4 multipliers (pixels) = 512 MACs;
    16 adder trees reduce the 8 channels per pixel -> [2,2,16].
    """
    return (a.reshape(4, TD).astype(np.int64) @ w8x16.astype(np.int64)).reshape(2, 2, TK)


def check_bits(v, bits, what):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if v.min() < lo or v.max() > hi:
        raise OverflowError(f"{what}: [{v.min()},{v.max()}] exceeds {bits} bits")


# --------------------------------------------------------------------------- #
# Layer geometry and analytic models
# --------------------------------------------------------------------------- #
def layer_geometry(H, D, K, s):
    N = H // s
    T = min(TILE_MAX, N)
    return dict(H_in=H, stride=s, D=D, K=K, N=N, T=T, n_tiles=(N // T) ** 2,
                T_in=s * (T - 1) + 3, positions=(T // 2) ** 2,
                n_dg=D // TD, n_kg=K // TK)


def analytic(g, fill):
    H, N, D, K, s = g["H_in"], g["N"], g["D"], g["K"], g["stride"]
    mac_dw, mac_pw = N * N * D * 9, N * N * D * K
    lat_tile = fill + g["positions"] * g["n_kg"]
    lat_total = lat_tile * g["n_tiles"] * g["n_dg"]
    tr = tc = s + 3                                           # DWC window per 2x2 block
    # Table II (PE-side accesses, loop order La, Tn=Tm=2)
    t2 = dict(dwc_act=tr * tc * D * (N * N) // (TN * TM), dwc_w=9 * D,
              pwc_act=N * N * D * K // TK, pwc_w=D * K)
    # Fig. 3 (external activation traffic; ifmap counted with its padding border)
    dwc_in, dwc_out, pwc_in, pwc_out = (H + 2) * (H + 2) * D, N * N * D, N * N * D, N * N * K
    base, fused = dwc_in + dwc_out + pwc_in + pwc_out, dwc_in + pwc_out
    return dict(mac_dw=mac_dw, mac_pw=mac_pw, lat_tile=lat_tile, lat_eq=lat_total,
                gops_eq=2 * (mac_dw + mac_pw) / lat_total, table2=t2,
                fig3_base=base, fig3_fused=fused, fig3_red=100 * (1 - fused / base))


# --------------------------------------------------------------------------- #
# Tile-level execution of one DSC layer
# --------------------------------------------------------------------------- #
def run_layer(x, wd, Kd, Bd, wp, Kp, Bp, s, F=16, fill=9, trace=None, layer_id=0):
    """x: [H,W,D] uint8 -> (ofmap [N,N,K] uint8, intermediate [N,N,D] uint8, counters)."""
    H, _, D = x.shape
    K = wp.shape[1]
    g = layer_geometry(H, D, K, s)
    N, T, T_in, n_dg, n_kg = g["N"], g["T"], g["T_in"], g["n_dg"], g["n_kg"]
    nt = N // T
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))                  # DMA inserts zeros at borders
    ofmap = np.zeros((N, N, K), np.uint8)
    inter = np.zeros((N, N, D), np.uint8)                     # debug tap only (not stored in HW)
    c = dict(cycles=0, dwc_fires=0, pwc_ops=0, ext_act_rd=0, ext_act_wr=0, ext_w_rd=0,
             ext_coef_rd=0, ifbuf_win_rd=0, inter_wr=0, inter_rd=0, psum_rmw=0,
             max_dwc_acc=0, max_pwc_acc=0)
    h = s + 3                                                  # window edge: 4 or 5

    for tr in range(nt):
        for tc in range(nt):
            psum = np.zeros((T // 2, T // 2, n_kg, 2, 2, TK), np.int64)   # 64 banks
            r0, c0 = s * tr * T, s * tc * T                    # padded-coordinate origin
            rows_real = len(range(max(r0 - 1, 0), min(r0 - 1 + T_in, H)))
            cols_real = len(range(max(c0 - 1, 0), min(c0 - 1 + T_in, H)))
            for dg in range(n_dg):
                ch = slice(dg * TD, (dg + 1) * TD)
                ifbuf = xp[r0:r0 + T_in, c0:c0 + T_in, ch]    # ifmap tile buffer
                wd8, kd8, bd8, wrow = wd[:, :, ch], Kd[ch], Bd[ch], wp[ch, :]
                c["ext_act_rd"] += rows_real * cols_real * TD
                c["ext_w_rd"] += 9 * TD + TD * K
                c["ext_coef_rd"] += 2 * TD
                base = c["cycles"]
                op = 0
                for pr in range(T // 2):
                    for pc in range(T // 2):
                        pos = pr * (T // 2) + pc
                        win = np.zeros((5, 5, TD), np.uint8)
                        win[:h, :h] = ifbuf[2 * s * pr:2 * s * pr + h, 2 * s * pc:2 * s * pc + h]
                        acc = dwc_engine(win, wd8, s)
                        check_bits(acc, 24, f"L{layer_id} DWC acc")
                        c["max_dwc_acc"] = max(c["max_dwc_acc"], int(np.abs(acc).max()))
                        a = nonconv(acc, kd8, bd8, F)
                        oy, ox = tr * T + 2 * pr, tc * T + 2 * pc
                        inter[oy:oy + 2, ox:ox + 2, ch] = a
                        c["dwc_fires"] += 1; c["ifbuf_win_rd"] += h * h * TD; c["inter_wr"] += 4 * TD
                        if trace is not None:
                            trace.append((base + pos * n_kg, "dwc_fire", layer_id, tr, tc, dg, pos, -1))
                        for kg in range(n_kg):
                            ps = pwc_engine(a, wrow[:, kg * TK:(kg + 1) * TK])
                            psum[pr, pc, kg] = ps if dg == 0 else psum[pr, pc, kg] + ps
                            check_bits(psum[pr, pc, kg], 32, f"L{layer_id} PWC psum")
                            c["max_pwc_acc"] = max(c["max_pwc_acc"], int(np.abs(psum[pr, pc, kg]).max()))
                            c["pwc_ops"] += 1; c["inter_rd"] += 4 * TD; c["psum_rmw"] += 4 * TK
                            ev = "pwc_result"
                            if dg == n_dg - 1:
                                ks = slice(kg * TK, (kg + 1) * TK)
                                ofmap[oy:oy + 2, ox:ox + 2, ks] = nonconv(psum[pr, pc, kg], Kp[ks], Bp[ks], F)
                                c["ext_act_wr"] += 4 * TK
                                ev = "ofmap_write"
                            if trace is not None:
                                trace.append((base + fill + op, ev, layer_id, tr, tc, dg, pos, kg))
                            op += 1
                c["cycles"] += fill + op                     # Eq. (1): FILL + (T/2)^2 * K/16
            c["ext_coef_rd"] += 2 * K                          # PWC-side K,B per tile
    return ofmap, inter, c, g


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_params(path):
    p = np.load(path)
    return {k: p[k] for k in p.files}


def layer_params(P, i):
    return (P[f"L{i}_wd"], P[f"L{i}_Kd"], P[f"L{i}_Bd"], P[f"L{i}_wp"],
            P[f"L{i}_Kp"], P[f"L{i}_Bp"], int(P[f"L{i}_stride"]))


def synthetic_params(rng):
    """Random weights/K/B with realistic magnitudes (selftest only)."""
    P, cin = {"frac_bits": np.int64(16)}, 32
    for i, (cout, s) in enumerate(DSC_CFG):
        P[f"L{i}_wd"] = rng.integers(-127, 128, (3, 3, cin)).astype(np.int8)
        P[f"L{i}_wp"] = rng.integers(-127, 128, (cin, cout)).astype(np.int8)
        P[f"L{i}_stride"] = np.int64(s)
        P[f"L{i}_Kd"] = rng.integers(-40, 400, cin).astype(np.int64)
        P[f"L{i}_Bd"] = rng.integers(-60 << 16, 60 << 16, cin).astype(np.int64)
        P[f"L{i}_Kp"] = rng.integers(-5, int(4e4 / np.sqrt(cin)), cout).astype(np.int64)
        P[f"L{i}_Bp"] = rng.integers(-60 << 16, 60 << 16, cout).astype(np.int64)
        cin = cout
    return P


def reference_layer(x, wd, Kd, Bd, wp, Kp, Bp, s, F):
    """Whole-layer integer model (int_model.py primitives) for one image."""
    a = nonconv(dwc_int(x[None], wd, s), Kd, Bd, F)
    y = nonconv(pwc_int(a, wp), Kp, Bp, F)
    return y[0], a[0]


def run_network(P, x0, fill, golden=None, img=0, trace_layer=None, trace=None):
    F = int(P["frac_bits"])
    x, rows = x0, []
    for i in range(len(DSC_CFG)):
        wd, Kd, Bd, wp, Kp, Bp, s = layer_params(P, i)
        tr = trace if (trace is not None and trace_layer == i) else None
        y, inter, c, g = run_layer(x, wd, Kd, Bd, wp, Kp, Bp, s, F, fill, tr, i)
        if golden is not None:
            ref_y, ref_a = golden[f"features.{i}.aq2"][img], golden[f"features.{i}.aq1"][img]
        else:
            ref_y, ref_a = reference_layer(x, wd, Kd, Bd, wp, Kp, Bp, s, F)
        ok_a, ok_y = np.array_equal(inter, ref_a), np.array_equal(y, ref_y)
        if not (ok_a and ok_y):
            bad = np.argwhere(y != ref_y)[:5] if not ok_y else np.argwhere(inter != ref_a)[:5]
            raise AssertionError(f"image {img} layer {i}: {'ofmap' if not ok_y else 'intermediate'} "
                                 f"mismatch, first (h,w,c): {bad.tolist()}")
        rows.append((g, c))
        x = y
    return x, rows


def report(rows, fill):
    print(f"\n{'L':>2} {'in':>5} {'s':>1} {'T':>2} {'tiles':>5} {'Lat_tile':>8} {'cycles':>7} "
          f"{'Eq.2':>6} {'GOPS':>7} {'paper':>7} {'PWC%':>6} {'DWC%':>6} {'Fig3 red%':>9}")
    out, tot_ops, tot_cyc, tot_b, tot_f = [], 0, 0, 0, 0
    for i, (g, c) in enumerate(rows):
        a = analytic(g, fill)
        ops = 2 * (a["mac_dw"] + a["mac_pw"])
        gops = ops / c["cycles"]
        pwc_u = 100 * c["pwc_ops"] / c["cycles"]
        dwc_u = 100 * c["dwc_fires"] / c["cycles"]
        tot_ops += ops; tot_cyc += c["cycles"]; tot_b += a["fig3_base"]; tot_f += a["fig3_fused"]
        print(f"{i:>2} {g['H_in']:>2}x{g['H_in']:<2} {g['stride']} {g['T']:>2} {g['n_tiles']:>5} "
              f"{a['lat_tile']:>8} {c['cycles']:>7} {a['lat_eq']:>6} {gops:7.1f} {PAPER_GOPS[i]:7.1f} "
              f"{pwc_u:6.1f} {dwc_u:6.1f} {a['fig3_red']:9.1f}")
        out.append({"layer": i, **g, **{k: v for k, v in a.items() if k != "table2"},
                    "table2": a["table2"], "counters": c, "gops": gops,
                    "paper_gops": PAPER_GOPS[i], "pwc_util": pwc_u, "dwc_util": dwc_u})
    avg_gops = np.mean([r["gops"] for r in out])
    print(f"\naverage throughput (mean of layers): {avg_gops:.2f} GOPS   (paper: 981.42)")
    print(f"network throughput (total ops / total cycles): {tot_ops / tot_cyc:.2f} GOPS, "
          f"{tot_cyc} cycles = {tot_cyc / 1e3:.2f} us @ 1 GHz")
    print(f"Fig.3 total activation-access reduction: {100 * (1 - tot_f / tot_b):.1f}%   (paper: 34.7%)")
    return out, {"avg_gops": avg_gops, "net_gops": tot_ops / tot_cyc, "total_cycles": tot_cyc,
                 "fig3_total_reduction": 100 * (1 - tot_f / tot_b)}


def buffer_sizing(out):
    sz = {
        "ifmap_tile_words_64b": max(r["T_in"] ** 2 for r in out),
        "intermediate_entries_256b": max(r["positions"] for r in out),
        "psum_words_32b": max(r["T"] ** 2 * r["K"] for r in out),
        "psum_bank_depth_64banks": max(r["positions"] * r["n_kg"] for r in out),
        "pwc_wbuf_rows_1024b": max(r["n_kg"] for r in out),
        "offline_dwc_coefs": max(r["D"] for r in out),
        "offline_pwc_coefs": max(r["K"] for r in out),
    }
    print("\nbuffer sizing derived from the schedule (per bank, before ping-pong):")
    for k, v in sz.items():
        print(f"  {k:<28} {v}")
    return sz


def make_plots(out, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping plots")
        return
    L = [r["layer"] for r in out]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(L, [r["gops"] for r in out], label="model")
    ax.plot(L, [r["paper_gops"] for r in out], "ro--", label="paper Fig. 13")
    ax.set_ylim(850, 1050); ax.set_xlabel("Layer"); ax.set_ylabel("Throughput (GOPS)")
    ax.set_xticks(L); ax.legend(); fig.tight_layout(); fig.savefig(f"{outdir}/fig13_throughput.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(L, [r["mac_dw"] + r["mac_pw"] for r in out], label="MAC operations")
    ax.set_ylabel("MACs"); ax.set_xlabel("Layer"); ax.set_xticks(L)
    ax2 = ax.twinx(); ax2.plot(L, [r["counters"]["cycles"] for r in out], "o-", color="tab:orange",
                               label="latency (ns @1GHz)")
    ax2.set_ylabel("Latency (ns)"); fig.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(f"{outdir}/fig10_mac_latency.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    w = 0.38
    ax.bar([l - w / 2 for l in L], [r["fig3_base"] for r in out], w, label="baseline")
    ax.bar([l + w / 2 for l in L], [r["fig3_fused"] for r in out], w, label="w/o intermediate access")
    ax.set_ylabel("Activation access count"); ax.set_xlabel("Layer"); ax.set_xticks(L)
    ax2 = ax.twinx(); ax2.plot(L, [r["fig3_red"] for r in out], "ro-", label="reduction %")
    ax2.set_ylim(0, 50); ax2.set_ylabel("Reduction (%)"); fig.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig3_access_reduction.png", dpi=150)
    print(f"plots saved to {outdir}/fig13_throughput.png, fig10_mac_latency.png, fig3_access_reduction.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="ckpt/edea_int_params.npz")
    ap.add_argument("--golden", default="ckpt/golden_samples.npz")
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--fill", type=int, default=9, help="pipeline fill cycles (paper: 9)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--trace-layer", type=int, default=None)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--outdir", default="ckpt")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    trace = [] if args.trace_layer is not None else None
    if args.selftest:
        rng = np.random.default_rng(1)
        P = synthetic_params(rng)
        x0 = rng.integers(0, 256, (32, 32, 32)).astype(np.uint8)
        _, rows = run_network(P, x0, args.fill, None, 0, args.trace_layer, trace)
        print("dataflow self-test: all 13 layers bit-exact vs int_model primitives (random weights)")
    else:
        P = load_params(args.params)
        G = np.load(args.golden)
        n = min(args.n_images, G["stem_aq"].shape[0])
        for img in range(n):
            _, rows = run_network(P, G["stem_aq"][img], args.fill, G, img,
                                  args.trace_layer if img == 0 else None, trace if img == 0 else None)
            print(f"image {img}: 13/13 layers bit-exact (intermediate + ofmap) vs golden")

    out, summary = report(rows, args.fill)
    sizing = buffer_sizing(out)

    cfg = [{k: r[k] for k in ("layer", "H_in", "stride", "D", "K", "N", "T", "n_tiles", "T_in",
                              "positions", "n_dg", "n_kg")} | {"cycles": r["counters"]["cycles"]}
           for r in out]
    with open(f"{args.outdir}/layer_cfg.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(f"{args.outdir}/dataflow_report.json", "w") as f:
        json.dump({"fill": args.fill, "summary": summary, "buffers": sizing, "layers": out}, f,
                  indent=2, default=int)
    if trace is not None:
        with open(f"{args.outdir}/trace_L{args.trace_layer}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["cycle", "event", "layer", "tile_r", "tile_c", "dg", "pos", "kg"])
            wr.writerows(sorted(trace))
        print(f"event trace -> {args.outdir}/trace_L{args.trace_layer}.csv ({len(trace)} events)")
    if args.plots:
        make_plots(out, args.outdir)
    print(f"saved {args.outdir}/layer_cfg.json and dataflow_report.json")


if __name__ == "__main__":
    main()
