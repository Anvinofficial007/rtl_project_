"""
export_vectors.py — Phase 5: export RTL test vectors.

Everything is packed into 64-bit words (8 lanes x 8 bit, lane 0 = bits [7:0]) so the
RTL needs a single external-memory word width.

WORD FORMATS
  activation  word[(dg*H + h)*W + w]        = channels 8*dg .. 8*dg+7 of pixel (h,w)   (uint8)
  DW weights  word[dg*9 + t], t = 3*i + j   = tap (i,j) of channels 8*dg .. 8*dg+7      (int8)
  PW weights  word[((dg*n_kg) + kg)*16 + ko]= W[8*dg .. 8*dg+7][16*kg + ko]              (int8)
              -> the 16 words of (dg,kg) form one 1024-bit PWC weight-buffer row
  Non-Conv    word[c] = {K_c[31:0], B_c[31:0]} (two's complement, sign-extended to 32 bit;
              the RTL uses the low KB_WIDTH bits)

OUTPUT (default ./vectors)
  ext_mem_params.hex        all weights + K/B for 13 layers (load at address 0)
  img<k>/L0_ifmap.hex       layer-0 input (stem output) of image k  (load at ACT_BASE)
  img<k>/L<l>_inter.hex     golden DWC->Non-Conv output of layer l   (debug tap)
  img<k>/L<l>_ofmap.hex     golden output of layer l (= input of layer l+1)
  img<k>/label.txt          true CIFAR-10 label
  params/L<l>_{dww,pww,ncd,ncp}.hex   per-layer slices for unit testbenches
  edea_layers.svh           SystemVerilog include: layer table + memory map
  memory_map.json           same information for Python / cocotb

Usage:
  python export_vectors.py                       # uses ckpt/edea_int_params.npz + golden_samples.npz
  python export_vectors.py --n-images 4
  python export_vectors.py --selftest            # synthetic data, no checkpoint needed
"""
import argparse
import json
import os

import numpy as np

from dataflow_model import DSC_CFG, TD, TK, layer_geometry, run_layer, synthetic_params, reference_layer

ALIGN = 16
ACT_ALIGN = 4096


def align(x, a):
    return (x + a - 1) // a * a


# --------------------------------------------------------------------------- #
# Packing / unpacking (the single source of truth for the RTL data layout)
# --------------------------------------------------------------------------- #
def lanes_to_words(lanes8):
    """[..., 8] uint8/int8 -> flat uint64 words, lane 0 in the LSBs."""
    a = np.ascontiguousarray(lanes8.astype(np.int64).astype(np.uint8)).reshape(-1, 8)
    return a.view("<u8").reshape(-1).astype(np.uint64)


def words_to_lanes(words, signed=False):
    b = np.ascontiguousarray(words.astype("<u8")).view(np.uint8).reshape(-1, 8)
    return b.view(np.int8) if signed else b


def pack_act(x):                       # [H,W,C] uint8 -> words [C/8 * H * W]
    H, W, C = x.shape
    return lanes_to_words(x.reshape(H, W, C // TD, TD).transpose(2, 0, 1, 3))


def unpack_act(words, H, W, C):
    return words_to_lanes(words).reshape(C // TD, H, W, TD).transpose(1, 2, 0, 3).reshape(H, W, C)


def pack_dww(wd):                      # [3,3,D] int8 -> [D/8 * 9]
    D = wd.shape[2]
    return lanes_to_words(wd.reshape(9, D // TD, TD).transpose(1, 0, 2))


def unpack_dww(words, D):
    return words_to_lanes(words, True).reshape(D // TD, 9, TD).transpose(1, 0, 2).reshape(3, 3, D)


def pack_pww(wp):                      # [D,K] int8 -> [D/8 * K/16 * 16]
    D, K = wp.shape
    return lanes_to_words(wp.reshape(D // TD, TD, K // TK, TK).transpose(0, 2, 3, 1))


def unpack_pww(words, D, K):
    return words_to_lanes(words, True).reshape(D // TD, K // TK, TK, TD).transpose(0, 3, 1, 2).reshape(D, K)


def pack_nc(Kc, Bc):
    for v, n in ((Kc, "K"), (Bc, "B")):
        assert v.min() >= -(1 << 31) and v.max() < (1 << 31), f"{n} exceeds 32 bit"
    k = Kc.astype(np.int64) & 0xFFFFFFFF
    b = Bc.astype(np.int64) & 0xFFFFFFFF
    return ((k << 32) | b).astype(np.uint64)


def unpack_nc(words):
    w = words.astype(np.uint64)
    k = (w >> np.uint64(32)).astype(np.uint32).view(np.int32).astype(np.int64)
    b = (w & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32).astype(np.int64)
    return k, b


def write_hex(path, words):
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v):016x}" for v in words))
        f.write("\n")


def read_hex(path):
    with open(path) as f:
        return np.array([int(l, 16) for l in f if l.strip()], dtype=np.uint64)


# --------------------------------------------------------------------------- #
# Memory map
# --------------------------------------------------------------------------- #
def build_map(P):
    layers, addr, H, D = [], 0, 32, int(P["L0_wd"].shape[2])
    for i in range(len(DSC_CFG)):
        K, s = P[f"L{i}_wp"].shape[1], int(P[f"L{i}_stride"])
        g = layer_geometry(H, D, K, s)
        e = dict(layer=i, **g)
        e["dww_base"] = addr; addr = align(addr + g["n_dg"] * 9, ALIGN)
        e["pww_base"] = addr; addr = align(addr + g["n_dg"] * g["n_kg"] * TK, ALIGN)
        e["ncd_base"] = addr; addr = align(addr + D, ALIGN)
        e["ncp_base"] = addr; addr = align(addr + K, ALIGN)
        layers.append(e)
        H, D = g["N"], K
    param_words = addr
    act = align(addr, ACT_ALIGN)
    act_base = act
    for e in layers:
        e["ifmap_base"] = act
        act = align(act + e["n_dg"] * e["H_in"] * e["H_in"], ALIGN)
    for e in layers:
        e["ofmap_base"] = layers[e["layer"] + 1]["ifmap_base"] if e["layer"] + 1 < len(layers) else act
    last = layers[-1]
    mem_words = align(last["ofmap_base"] + (last["K"] // TD) * last["N"] * last["N"], ACT_ALIGN)
    return layers, dict(param_words=param_words, act_base=act_base, mem_words=mem_words)


def build_param_image(P, layers, mm):
    mem = np.zeros(mm["param_words"], np.uint64)
    for e in layers:
        i = e["layer"]
        for base, words in ((e["dww_base"], pack_dww(P[f"L{i}_wd"])),
                            (e["pww_base"], pack_pww(P[f"L{i}_wp"])),
                            (e["ncd_base"], pack_nc(P[f"L{i}_Kd"], P[f"L{i}_Bd"])),
                            (e["ncp_base"], pack_nc(P[f"L{i}_Kp"], P[f"L{i}_Bp"]))):
            mem[base:base + len(words)] = words
    return mem


def write_svh(path, layers, mm, kb_width, frac):
    def arr(key):
        return "'{" + ", ".join(str(int(e[key])) for e in layers) + "}"
    keys = [("L_H_IN", "H_in"), ("L_STRIDE", "stride"), ("L_D", "D"), ("L_K", "K"), ("L_N", "N"),
            ("L_T", "T"), ("L_NTILES", "n_tiles"), ("L_TIN", "T_in"), ("L_NDG", "n_dg"),
            ("L_NKG", "n_kg"), ("L_IFMAP_BASE", "ifmap_base"), ("L_OFMAP_BASE", "ofmap_base"),
            ("L_DWW_BASE", "dww_base"), ("L_PWW_BASE", "pww_base"),
            ("L_NCD_BASE", "ncd_base"), ("L_NCP_BASE", "ncp_base"), ("L_CYCLES_REF", "cycles_ref")]
    lines = ["// edea_layers.svh -- AUTO-GENERATED by export_vectors.py, do not edit",
             "// addresses are in 64-bit words",
             f"localparam int N_LAYERS    = {len(layers)};",
             f"localparam int MEM_WORDS   = {mm['mem_words']};",
             f"localparam int PARAM_WORDS = {mm['param_words']};",
             f"localparam int ACT_BASE    = {mm['act_base']};",
             f"localparam int KB_WIDTH    = {kb_width};",
             f"localparam int KB_FRAC     = {frac};"]
    lines += [f"localparam int {n:<13} [N_LAYERS] = {arr(k)};" for n, k in keys]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# End-to-end check: run all layers reading ONLY from the packed memory image
# --------------------------------------------------------------------------- #
def emulate_from_memory(mem, layers, F, golden_inter, golden_out):
    for e in layers:
        H, D, K, N = e["H_in"], e["D"], e["K"], e["N"]
        x = unpack_act(mem[e["ifmap_base"]:e["ifmap_base"] + e["n_dg"] * H * H], H, H, D)
        wd = unpack_dww(mem[e["dww_base"]:e["dww_base"] + e["n_dg"] * 9], D)
        wp = unpack_pww(mem[e["pww_base"]:e["pww_base"] + e["n_dg"] * e["n_kg"] * TK], D, K)
        Kd, Bd = unpack_nc(mem[e["ncd_base"]:e["ncd_base"] + D])
        Kp, Bp = unpack_nc(mem[e["ncp_base"]:e["ncp_base"] + K])
        y, inter, _, _ = run_layer(x, wd, Kd, Bd, wp, Kp, Bp, e["stride"], F)
        assert np.array_equal(inter, golden_inter[e["layer"]]), f"L{e['layer']} intermediate mismatch"
        assert np.array_equal(y, golden_out[e["layer"]]), f"L{e['layer']} ofmap mismatch"
        out_words = pack_act(y)
        mem[e["ofmap_base"]:e["ofmap_base"] + len(out_words)] = out_words


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="ckpt/edea_int_params.npz")
    ap.add_argument("--golden", default="ckpt/golden_samples.npz")
    ap.add_argument("--layer-cfg", default="ckpt/layer_cfg.json")
    ap.add_argument("--out", default="vectors")
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        rng = np.random.default_rng(3)
        P = synthetic_params(rng)
        x = rng.integers(0, 256, (32, 32, 32)).astype(np.uint8)
        G = {"stem_aq": x[None], "label": np.array([0])}
        for i in range(len(DSC_CFG)):
            wd, Kd, Bd, wp, Kp, Bp = (P[f"L{i}_{n}"] for n in ("wd", "Kd", "Bd", "wp", "Kp", "Bp"))
            y, a = reference_layer(x, wd, Kd, Bd, wp, Kp, Bp, int(P[f"L{i}_stride"]), 16)
            G[f"features.{i}.aq1"], G[f"features.{i}.aq2"] = a[None], y[None]
            x = y
        args.n_images = 1
    else:
        P = dict(np.load(args.params))
        G = dict(np.load(args.golden))
    F = int(P["frac_bits"])
    kb_width = int(P.get("kb_width", 24))

    layers, mm = build_map(P)
    cyc = {}
    if os.path.exists(args.layer_cfg) and not args.selftest:
        cyc = {c["layer"]: c["cycles"] for c in json.load(open(args.layer_cfg))}
    for e in layers:
        e["cycles_ref"] = cyc.get(e["layer"], 0)

    os.makedirs(f"{args.out}/params", exist_ok=True)
    pmem = build_param_image(P, layers, mm)
    write_hex(f"{args.out}/ext_mem_params.hex", pmem)
    for e in layers:
        i = e["layer"]
        for tag, base, n in (("dww", e["dww_base"], e["n_dg"] * 9),
                             ("pww", e["pww_base"], e["n_dg"] * e["n_kg"] * TK),
                             ("ncd", e["ncd_base"], e["D"]), ("ncp", e["ncp_base"], e["K"])):
            write_hex(f"{args.out}/params/L{i}_{tag}.hex", pmem[base:base + n])

    n_img = min(args.n_images, G["stem_aq"].shape[0])
    for k in range(n_img):
        d = f"{args.out}/img{k}"
        os.makedirs(d, exist_ok=True)
        write_hex(f"{d}/L0_ifmap.hex", pack_act(G["stem_aq"][k]))
        g_inter = [G[f"features.{i}.aq1"][k] for i in range(len(layers))]
        g_out = [G[f"features.{i}.aq2"][k] for i in range(len(layers))]
        for i in range(len(layers)):
            write_hex(f"{d}/L{i}_inter.hex", pack_act(g_inter[i]))
            write_hex(f"{d}/L{i}_ofmap.hex", pack_act(g_out[i]))
        with open(f"{d}/label.txt", "w") as f:
            f.write(f"{int(G['label'][k])}\n")

        # round-trip: re-read the hex files and execute the whole network from them
        mem = np.zeros(mm["mem_words"], np.uint64)
        mem[:mm["param_words"]] = read_hex(f"{args.out}/ext_mem_params.hex")
        l0 = read_hex(f"{d}/L0_ifmap.hex")
        mem[layers[0]["ifmap_base"]:layers[0]["ifmap_base"] + len(l0)] = l0
        emulate_from_memory(mem, layers, F, g_inter, g_out)
        print(f"img{k}: hex round-trip + 13-layer execution from memory image: bit-exact")

    write_svh(f"{args.out}/edea_layers.svh", layers, mm, kb_width, F)
    with open(f"{args.out}/memory_map.json", "w") as f:
        json.dump({**mm, "kb_width": kb_width, "frac_bits": F, "word_bits": 64,
                   "layers": layers}, f, indent=2, default=int)

    print(f"\nmemory map (64-bit words): params 0..{mm['param_words'] - 1}, "
          f"activations from {mm['act_base']}, total {mm['mem_words']} words "
          f"({mm['mem_words'] * 8 / 1024:.0f} KiB)")
    print(f"{'L':>2} {'ifmap':>8} {'ofmap':>8} {'dww':>8} {'pww':>8} {'ncd':>8} {'ncp':>8}")
    for e in layers:
        print(f"{e['layer']:>2} {e['ifmap_base']:>8} {e['ofmap_base']:>8} {e['dww_base']:>8} "
              f"{e['pww_base']:>8} {e['ncd_base']:>8} {e['ncp_base']:>8}")
    print(f"\nKB_WIDTH = {kb_width} (Q{kb_width - F}.{F})   vectors written to {args.out}/")


if __name__ == "__main__":
    main()
