"""Train the DeepLabV3+ decoder on top of the frozen Qwen3-VL-4B vision tower, on the
real, fully-extracted AI4Mars dataset (see dataset_ai4mars.py for the camera subsets
and their verified label encodings).

Only the decoder is trained; the vision encoder stays frozen throughout (Stage 1 scope).

By default this combines ALL 4 compatible camera subsets (msl_mcam, msl_ncam, mer,
m2020_nav -- ~36.8K pairs total) into one training pool; pass --subsets to train on
just one or a few instead.

Examples:
    python train_segmentation.py --epochs 20                       # whole dataset
    python train_segmentation.py --subsets msl_ncam --epochs 20    # one subset only
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dataset_ai4mars import (
    AI4MarsSegDataset,
    CAMERA_SUBSETS,
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    collate_images_labels,
    find_all_pairs,
    split_pairs,
)
from deeplab_decoder import DeepLabV3PlusDecoder
from vision_encoder import Qwen3VLVisionEncoder

DEFAULT_DATASET_ROOT = (
    "/media/nahar4/2e670039-b303-4e14-b517-49b4319b069d/AI4mars/extracted/"
    "ai4mars-dataset-merged-0.6"
)


def compute_class_weights(pairs, sample_size: int | None = None) -> torch.Tensor:
    sample = pairs if sample_size is None else random.Random(0).sample(pairs, min(sample_size, len(pairs)))
    counts = torch.zeros(256, dtype=torch.float64)  # wide enough to safely index any real label value
    for _, lbl_path, raw_ignore_index, _scene_id in sample:
        arr = np.array(Image.open(lbl_path))
        vals, cnts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            if v == raw_ignore_index:
                continue
            counts[v] += c
    counts = counts[:NUM_CLASSES]
    freq = counts / counts.sum()
    weights = 1.0 / torch.clamp(freq, min=1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return weights.float()


def soft_dice_loss(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_index: int,
    weight: torch.Tensor | None = None, eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-class soft Dice loss (1 - per-class Dice, optionally class-weighted the
    same way as the CE loss) -- ignored pixels are excluded from both intersection and
    union rather than being mapped to a real class, so they can't inflate any class's
    score. Directly optimizes toward IoU, unlike plain cross-entropy."""
    valid = labels != ignore_index
    probs = torch.softmax(logits, dim=1)
    labels_safe = torch.where(valid, labels, torch.zeros_like(labels))
    targets = F.one_hot(labels_safe, num_classes).permute(0, 3, 1, 2).float()

    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    targets = targets * valid_f

    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    loss_per_class = 1 - dice_per_class

    if weight is not None:
        return (loss_per_class * weight).sum() / weight.sum()
    return loss_per_class.mean()


@torch.no_grad()
def evaluate(encoder, decoder, loader, device, ignore_index: int) -> dict:
    decoder.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    for images, labels in loader:
        labels = labels.to(device)
        feats = encoder(images)
        logits = decoder(feats["low_level"].float(), feats["high_level"].float(), output_size=labels.shape[-2:])
        preds = logits.argmax(dim=1)

        valid = labels != ignore_index
        p, t = preds[valid], labels[valid]
        idx = t * NUM_CLASSES + p
        binc = torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)
        confusion += binc.view(NUM_CLASSES, NUM_CLASSES)

    ious = []
    for c in range(NUM_CLASSES):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))

    pixel_acc = confusion.diag().sum().item() / confusion.sum().item()
    decoder.train()
    return {
        "per_class_iou": dict(zip(CLASS_NAMES, ious)),
        "mean_iou": float(np.nanmean(ious)),
        "pixel_accuracy": pixel_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument(
        "--subsets", nargs="+", choices=sorted(CAMERA_SUBSETS), default=sorted(CAMERA_SUBSETS),
        help="camera subsets to combine into one training pool (default: all 4)",
    )
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--warmup-steps", type=int, default=-1,
        help="linear LR warmup steps before the cosine decay; -1 = auto (3%% of total steps, capped at 500)",
    )
    ap.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm for the decoder (0 = disabled)")
    ap.add_argument(
        "--dice-weight", type=float, default=0.0,
        help="weight of the soft Dice loss term added to the weighted CE loss (0 = CE only, the proven baseline "
             "recipe -- a dice_weight=1.0 + --augment run plateaued at 0.75 mean IoU, below the 0.765 CE-only "
             "baseline, so this is opt-in, not default)",
    )
    ap.add_argument(
        "--augment", action="store_true",
        help="enable train-time crop/flip/color-jitter augmentation (off by default -- see --dice-weight help; "
             "val split is never augmented)",
    )
    ap.add_argument("--aspp-channels", type=int, default=256, help="ASPP output / decoder fuse channel width")
    ap.add_argument(
        "--low-level-proj-channels", type=int, default=48,
        help="channel width of the projected low-level skip features before fusion",
    )
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epoch(s); >0 = stop early, for smoke runs")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", default=None, help="default: runs/<subsets>_stage1")
    ap.add_argument("--resume", default=None, help="path to a decoder .pt checkpoint to initialize from")
    args = ap.parse_args()
    if args.out_dir is None:
        tag = "all" if sorted(args.subsets) == sorted(CAMERA_SUBSETS) else "_".join(args.subsets)
        args.out_dir = f"runs/{tag}_stage1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_all_pairs(args.dataset_root, args.subsets)
    train_pairs, val_pairs = split_pairs(pairs, val_fraction=args.val_fraction)
    print(f"subsets={args.subsets} ignore_index={IGNORE_INDEX} (canonical, after per-subset remap)")
    print(f"train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")

    print("computing class weights from training split ...")
    class_weights = compute_class_weights(train_pairs, sample_size=4000).to(device)
    print("class weights:", dict(zip(CLASS_NAMES, class_weights.tolist())))

    train_ds = AI4MarsSegDataset(train_pairs, image_size=args.image_size, augment=args.augment)
    val_ds = AI4MarsSegDataset(val_pairs, image_size=args.image_size, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_images_labels, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_images_labels,
    )

    print("loading frozen Qwen3-VL-4B vision tower ...")
    encoder = Qwen3VLVisionEncoder(device=device, dtype=torch.bfloat16)
    decoder = DeepLabV3PlusDecoder(
        num_classes=NUM_CLASSES,
        high_level_channels=encoder.hidden_size,
        low_level_channels=encoder.hidden_size,
        aspp_out_channels=args.aspp_channels,
        low_level_proj_channels=args.low_level_proj_channels,
    ).to(device)
    if args.resume:
        decoder.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed decoder weights from {args.resume}")
        print("note: optimizer/scheduler state is NOT resumed (wasn't saved) -- "
              "this continuation starts with a fresh AdamW + a new cosine schedule "
              "over --epochs; consider a lower --lr than the original run's peak "
              "so the fresh schedule doesn't disrupt already-learned weights too hard")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    total_steps = (len(train_loader) * args.epochs) if not args.max_steps else args.max_steps
    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else min(500, max(1, total_steps * 3 // 100))
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1)),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    step = 0
    history = []
    best_mean_iou = -1.0
    t0 = time.time()
    for epoch in range(args.epochs):
        for images, labels in train_loader:
            labels = labels.to(device)
            feats = encoder(images)
            logits = decoder(feats["low_level"].float(), feats["high_level"].float(), output_size=labels.shape[-2:])
            loss_ce = criterion(logits, labels)
            loss_dice = soft_dice_loss(logits, labels, NUM_CLASSES, IGNORE_INDEX, weight=class_weights)
            loss = loss_ce + args.dice_weight * loss_dice

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 20 == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"(ce {loss_ce.item():.4f} dice {loss_dice.item():.4f}) lr {lr_now:.2e} ({elapsed:.0f}s elapsed)"
                )
                history.append({
                    "step": step, "epoch": epoch, "loss": loss.item(),
                    "loss_ce": loss_ce.item(), "loss_dice": loss_dice.item(), "lr": lr_now,
                })

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

        metrics = evaluate(encoder, decoder, val_loader, device, IGNORE_INDEX)
        print(f"epoch {epoch} val metrics: {json.dumps(metrics, indent=2)}")
        history.append({"epoch": epoch, "val_metrics": metrics})

        if metrics["mean_iou"] > best_mean_iou:
            best_mean_iou = metrics["mean_iou"]
            torch.save(decoder.state_dict(), out_dir / "decoder_best.pt")
            print(f"epoch {epoch}: new best mean_iou {best_mean_iou:.4f}, saved decoder_best.pt")

        # Written after every epoch (not just at the end) so an interruption never loses
        # the per-epoch trend -- this is what's needed to judge whether more epochs help.
        torch.save(decoder.state_dict(), out_dir / "decoder_last.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(
                {"history": history, "best_mean_iou": best_mean_iou, "completed_epochs": epoch + 1, "args": vars(args)},
                f, indent=2,
            )

    final_metrics = evaluate(encoder, decoder, val_loader, device, IGNORE_INDEX)
    print("final val metrics:", json.dumps(final_metrics, indent=2))

    torch.save(decoder.state_dict(), out_dir / "decoder_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(
            {
                "history": history, "final_metrics": final_metrics, "best_mean_iou": best_mean_iou,
                "completed_epochs": args.epochs, "args": vars(args),
            },
            f, indent=2,
        )
    print(f"saved decoder_final.pt (best_mean_iou={best_mean_iou:.4f} in decoder_best.pt) + history to {out_dir}")


if __name__ == "__main__":
    main()
