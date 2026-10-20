"""
models.py — MobileNetV1 for CIFAR-10, structured to match the EDEA paper.

Structure (32x32 input):
  stem : conv3x3 (3->32, s1) + BN + ReLU            -> runs in software later
  DSC layers 0..12 (each: DW3x3 + BN + ReLU, PW1x1 + BN + ReLU)  -> RTL
  head : global avg-pool + FC(1024->10)              -> runs in software later

Design rules that matter for the later phases:
  * plain ReLU (not ReLU6), as in the paper
  * no conv bias (BN provides it; folded into Non-Conv k,b in Phase 3)
  * module names are fixed: features[i].dw / bn1 / pw / bn2, so the QAT and
    folding scripts can find every layer by name.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# (out_channels, stride) for DSC layers 0..12 — matches the table in the plan
DSC_CFG = [
    (64, 1),            # 0  : 32x32, D=32  -> K=64
    (128, 2),           # 1  : 32x32 -> 16x16
    (128, 1),           # 2
    (256, 2),           # 3  : 16x16 -> 8x8
    (256, 1),           # 4
    (512, 2),           # 5  : 8x8 -> 4x4
    (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),   # 6..10
    (1024, 2),          # 11 : 4x4 -> 2x2
    (1024, 1),          # 12
]


class DSCBlock(nn.Module):
    """One depthwise-separable layer = one pass through the accelerator."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.stride = stride
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))   # -> Non-Conv unit #1 in HW
        x = F.relu(self.bn2(self.pw(x)))   # -> output requant in HW
        return x


class MobileNetV1CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, stem_ch: int = 32):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, stem_ch, 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(stem_ch)

        layers, cin = [], stem_ch
        for cout, s in DSC_CFG:
            layers.append(DSCBlock(cin, cout, s))
            cin = cout
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


@torch.no_grad()
def layer_report(model: MobileNetV1CIFAR, verbose: bool = True):
    """Print per-DSC-layer shapes and MAC counts; check against the plan's table."""
    rows, x = [], torch.zeros(1, 3, 32, 32)
    x = F.relu(model.stem_bn(model.stem_conv(x)))
    for i, blk in enumerate(model.features):
        R, D = x.shape[2], x.shape[1]
        y = blk(x)
        N, K = y.shape[2], y.shape[1]
        mac_dw, mac_pw = N * N * D * 9, N * N * D * K
        rows.append(dict(layer=i, R=R, s=blk.stride, N=N, D=D, K=K,
                         mac_dw=mac_dw, mac_pw=mac_pw))
        x = y
    if verbose:
        print(f"{'L':>2} {'ifmap':>7} {'s':>2} {'ofmap':>7} {'D':>5} {'K':>5} "
              f"{'MAC_dw':>9} {'MAC_pw':>10} {'total':>10}")
        for r in rows:
            print(f"{r['layer']:>2} {r['R']:>3}x{r['R']:<3} {r['s']:>2} "
                  f"{r['N']:>3}x{r['N']:<3} {r['D']:>5} {r['K']:>5} "
                  f"{r['mac_dw']:>9,} {r['mac_pw']:>10,} {r['mac_dw'] + r['mac_pw']:>10,}")
        tot = sum(r["mac_dw"] + r["mac_pw"] for r in rows)
        params = sum(p.numel() for p in model.parameters())
        print(f"DSC total MACs: {tot:,}   model params: {params:,}")
    return rows


# Expected values (from the plan, Section 2.2) — used as a self-check.
EXPECTED_TOTAL_MACS = [2392064, 2244608, 4489216, 2170880, 4341760, 2134016,
                       4268032, 4268032, 4268032, 4268032, 4268032, 2115584, 4231168]

if __name__ == "__main__":
    m = MobileNetV1CIFAR()
    rows = layer_report(m)
    got = [r["mac_dw"] + r["mac_pw"] for r in rows]
    assert got == EXPECTED_TOTAL_MACS, "layer shapes do not match the EDEA table!"
    print("Shape/MAC check vs EDEA table: PASS")
