"""
train_qat.py — Phase 2: INT8 quantization-aware fine-tuning with LSQ.

Usage:
  python train_qat.py --float-ckpt ckpt/float_best.pth            # full run (60 epochs)
  python train_qat.py --float-ckpt ckpt/float_best.pth --epochs 1 # smoke test
  python train_qat.py --resume ckpt/qat_last.pth

Flow:
  1. load float weights -> evaluate float accuracy
  2. init weight steps from weights, calibrate activation steps on data
     -> evaluate "PTQ" accuracy (quantized, before any fine-tuning)
  3. fine-tune (weights, BN and all step sizes learn); BN stats frozen for the
     last --freeze-bn epochs so exported mu/sigma match what the net was tuned on
  4. save qat_best.pth, log, quantizer report and activation stats (zero %)

Outputs (in --out):
  qat_best.pth, qat_last.pth, qat_log.csv, qat_summary.json
"""
import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn

from models import MobileNetV1CIFAR
from qmodels import (QMobileNetV1CIFAR, load_float_weights, init_weight_steps,
                     calibrate_activations, set_bn_eval, quant_report,
                     activation_stats, lsq_modules)
from train_float import get_loaders, evaluate, set_seed, lr_at


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--float-ckpt", default="ckpt/float_best.pth")
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./ckpt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--freeze-bn", type=int, default=10, help="freeze BN stats for last N epochs")
    p.add_argument("--calib-batches", type=int, default=20)
    p.add_argument("--act-init", choices=["percentile", "lsq"], default="percentile")
    p.add_argument("--per-tensor", action="store_true", help="per-tensor weight steps")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", default="")
    return p.parse_args()


def param_groups(model, wd):
    """No weight decay on BN params, biases and LSQ step sizes."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim <= 1 or name.endswith(".step"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def train_one_epoch(model, loader, opt, scaler, crit, epoch, args, dev, freeze_bn):
    model.train()
    if freeze_bn:
        set_bn_eval(model)
    n, correct, loss_sum, iters = 0, 0, 0.0, len(loader)
    for i, (x, y) in enumerate(loader):
        lr = lr_at(epoch + i / iters, args)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, enabled=args.amp):
            out = model(x)
            loss = crit(out, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, 100.0 * correct / n, lr


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        args.amp = False
    print(f"device: {dev}")
    train_loader, test_loader = get_loaders(args)

    model = QMobileNetV1CIFAR(per_channel=not args.per_tensor).to(dev)
    summary = {}

    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        summary = ck.get("summary", {})
        print(f"resumed from epoch {ck['epoch']}, best {best_acc:.2f}%")
    else:
        # 1. float baseline
        fstate = torch.load(args.float_ckpt, map_location=dev, weights_only=False)["model"]
        fmodel = MobileNetV1CIFAR().to(dev)
        fmodel.load_state_dict(fstate)
        _, float_acc = evaluate(fmodel, test_loader, dev)
        print(f"float accuracy: {float_acc:.2f}%")
        del fmodel

        # 2. quantizer init + PTQ accuracy
        load_float_weights(model, fstate)
        init_weight_steps(model)
        calibrate_activations(model, train_loader, dev, args.calib_batches, args.act_init)
        _, ptq_acc = evaluate(model, test_loader, dev)
        print(f"INT8 accuracy after calibration (PTQ, no fine-tuning): {ptq_acc:.2f}%")
        summary = {"float_acc": float_acc, "ptq_acc": ptq_acc}
        start_epoch, best_acc = 0, 0.0

    opt = torch.optim.SGD(param_groups(model, args.wd), lr=args.lr, momentum=0.9, nesterov=True)
    if args.resume and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    crit = nn.CrossEntropyLoss()

    log_path = os.path.join(args.out, "qat_log.csv")
    if not args.resume:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "bn_frozen", "train_loss", "train_acc",
                                    "test_loss", "test_acc", "time_s"])

    # 3. fine-tune
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        freeze = epoch >= args.epochs - args.freeze_bn
        tr_loss, tr_acc, lr = train_one_epoch(model, train_loader, opt, scaler, crit,
                                              epoch, args, dev, freeze)
        te_loss, te_acc = evaluate(model, test_loader, dev)
        dt = time.time() - t0

        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                 "best_acc": max(best_acc, te_acc), "args": vars(args), "summary": summary}
        torch.save(state, os.path.join(args.out, "qat_last.pth"))
        # only checkpoints with frozen BN stats are eligible as "best" when freezing is used,
        # so the exported mu/sigma are the ones the weights were tuned against
        eligible = freeze or args.freeze_bn == 0
        if eligible and te_acc > best_acc:
            best_acc = te_acc
            torch.save(state, os.path.join(args.out, "qat_best.pth"))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.5f}", int(freeze), f"{tr_loss:.4f}",
                                    f"{tr_acc:.2f}", f"{te_loss:.4f}", f"{te_acc:.2f}", f"{dt:.1f}"])
        print(f"ep {epoch:3d} | lr {lr:.5f} | bnF {int(freeze)} | train {tr_loss:.3f}/{tr_acc:5.2f}% "
              f"| test {te_loss:.3f}/{te_acc:5.2f}% | best {best_acc:5.2f}% | {dt:.0f}s")

    # 4. report on the best checkpoint
    best = torch.load(os.path.join(args.out, "qat_best.pth"), map_location=dev, weights_only=False)
    model.load_state_dict(best["model"])
    _, qat_acc = evaluate(model, test_loader, dev)
    print("\n=== Quantizer step sizes ===")
    quant_report(model)

    stats = activation_stats(model, test_loader, dev)
    print("\n=== Activation codes on test set (zero % ~ paper Fig. 11) ===")
    for k, v in stats.items():
        print(f"{k:<20} zero {v['zero_pct']:6.2f}%   saturated(255) {v['sat_pct']:6.3f}%")

    summary.update({"qat_acc": qat_acc, "activation_stats": stats,
                    "steps": {n: m.step.detach().cpu().tolist() if m.channels == 1 else
                              [m.step.min().item(), m.step.max().item()]
                              for n, m in lsq_modules(model)}})
    best["summary"] = summary
    torch.save(best, os.path.join(args.out, "qat_best.pth"))
    with open(os.path.join(args.out, "qat_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nfloat {summary.get('float_acc', float('nan')):.2f}% | "
          f"PTQ {summary.get('ptq_acc', float('nan')):.2f}% | QAT INT8 {qat_acc:.2f}%")


if __name__ == "__main__":
    main()
