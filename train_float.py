"""
train_float.py — Phase 1: train the float MobileNetV1 on CIFAR-10.

Usage:
  python models.py                      # shape/MAC self-check first
  python train_float.py                 # full run (200 epochs)
  python train_float.py --epochs 2      # smoke test
  python train_float.py --resume ckpt/float_last.pth

Outputs (in --out, default ./ckpt):
  float_best.pth   best test accuracy  (input to Phase 2 QAT)
  float_last.pth   last epoch (for resuming)
  float_log.csv    epoch, lr, train_loss, train_acc, test_loss, test_acc, time
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from models import MobileNetV1CIFAR, layer_report

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./ckpt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=5, help="linear warm-up epochs")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="mixed precision (GPU)")
    p.add_argument("--resume", default="")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(args):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_set = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=test_tf)
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=pin)
    return train_loader, test_loader


def param_groups(model, wd):
    """Weight decay on conv/linear weights only; none on BN params and biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at(epoch_float, args):
    """Linear warm-up, then cosine decay to 0 (per-iteration)."""
    if epoch_float < args.warmup:
        return float(args.lr * epoch_float / args.warmup)
    t = (epoch_float - args.warmup) / max(1, args.epochs - args.warmup)
    return float(0.5 * args.lr * (1 + np.cos(np.pi * t)))


def train_one_epoch(model, loader, opt, scaler, crit, epoch, args, dev):
    model.train()
    n, correct, loss_sum = 0, 0, 0.0
    iters = len(loader)
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


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    crit = nn.CrossEntropyLoss()
    n, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss_sum += crit(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, 100.0 * correct / n


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        args.amp = False
    print(f"device: {dev}")

    model = MobileNetV1CIFAR().to(dev)
    layer_report(model.cpu()); model.to(dev)

    train_loader, test_loader = get_loaders(args)
    opt = torch.optim.SGD(param_groups(model, args.wd), lr=args.lr,
                          momentum=0.9, nesterov=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_epoch, best_acc = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        print(f"resumed from epoch {ck['epoch']}, best {best_acc:.2f}%")

    log_path = os.path.join(args.out, "float_log.csv")
    if not args.resume:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc",
                                    "test_loss", "test_acc", "time_s"])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc, lr = train_one_epoch(model, train_loader, opt, scaler, crit, epoch, args, dev)
        te_loss, te_acc = evaluate(model, test_loader, dev)
        dt = time.time() - t0

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "epoch": epoch, "best_acc": max(best_acc, te_acc), "args": vars(args)}
        torch.save(state, os.path.join(args.out, "float_last.pth"))
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(state, os.path.join(args.out, "float_best.pth"))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.5f}", f"{tr_loss:.4f}", f"{tr_acc:.2f}",
                                    f"{te_loss:.4f}", f"{te_acc:.2f}", f"{dt:.1f}"])
        print(f"ep {epoch:3d} | lr {lr:.4f} | train {tr_loss:.3f}/{tr_acc:5.2f}% "
              f"| test {te_loss:.3f}/{te_acc:5.2f}% | best {best_acc:5.2f}% | {dt:.0f}s")

    print(f"Done. Best float test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
