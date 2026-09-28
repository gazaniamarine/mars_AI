# Mars terrain segmentation — training log

Setup: DeepLabV3+ decoder (ASPP + low-level skip) trained on top of a **frozen**
Qwen3-VL-4B vision tower (encoder weights never updated — a hard constraint, not
just Stage 1 scope). 4 classes (soil, bedrock, sand, big_rock), ~37K AI4Mars pairs
across 4 camera subsets, scene-disjoint train/val split.

## Runs

| Run (`runs/`) | Epochs | Change vs. previous | mean IoU | big_rock IoU |
|---|---|---|---|---|
| `all_stage1` | 50 | Baseline: weighted cross-entropy only | 0.765 | 0.611 |
| `all_stage1_v2` | 50 | + augmentation (flip/crop-scale/color-jitter) + Dice loss (weight 1.0) | 0.745 ↓ | 0.599 ↓ |
| `all_stage1_v2_2` | 90 | Same recipe, longer schedule (check if it just needed more time) | 0.751 ↓ | 0.610 |
| `all_stage1_wide` | 50 | Reverted to CE-only loss, augmentation off; **added** LR warmup + grad clipping + wider decoder (ASPP 256→384 ch, low-level proj 48→96 ch) | **0.768 ↑ (new best)** | **0.630 ↑ (new best)** |

## What happened, in order

1. **Baseline (`all_stage1`)**: plain class-weighted CE, no augmentation. 0.765 mean IoU.
   Weakest class by far: `big_rock` (0.611) — rarest/smallest class, hardest edges.

2. **Tried augmentation + Dice loss (`all_stage1_v2`)**: added synchronized random
   crop/scale-jitter + horizontal flip + color jitter to the training data, and added
   a class-weighted soft Dice loss term (`CE + 1.0×Dice`) since Dice targets IoU
   directly. Result regressed across every class (0.765 → 0.745).

3. **Diagnosed the regression**: pulled `history.json` and checked step-level losses —
   no bug, no NaNs/divergence, loss values were sane throughout. The real cause: the
   per-epoch IoU curve was still climbing at epoch 49, never plateaued, unlike the
   baseline which flattened out by ~epoch 32. Looked like the harder augmented task
   just needed more training time.

4. **Re-ran with 90 epochs (`all_stage1_v2_2`)** to test that theory. This time it did
   plateau (flat ~0.748–0.754 from epoch ~70 on) — but *below* the baseline's 0.765,
   not above it. So it wasn't a convergence-speed problem after all: augmentation +
   Dice, as configured, has a genuinely lower ceiling in this setup.

5. **Root cause**: the encoder is frozen, so only the decoder can adapt. Normal
   segmentation training benefits from augmentation because the *whole* network learns
   to be invariant to the perturbations. Here, a relatively shallow decoder was being
   asked to become invariant to scale/color changes on top of fixed, generic
   (non-terrain-tuned) features that can't adjust — a much weaker lever than usual, and
   in this case a net negative.

6. **Response, keeping the encoder frozen as required (`all_stage1_wide`)**:
   - Reverted `--dice-weight` and `--augment` to off-by-default (opt-in flags now,
     instead of on-by-default) so the default recipe matches the proven baseline again.
   - Added linear LR warmup (first 3% of steps) before the cosine decay, and gradient
     clipping (max norm 1.0) — cheap stability improvements for the BatchNorm-heavy
     ASPP block, independent of the loss/augmentation question.
   - Since the encoder can't be touched, the only way to extract more from its fixed
     features is a **wider decoder** — exposed ASPP channel width and low-level
     projection width as CLI flags and doubled both (256→384, 48→96).
   - Result: **0.768 mean IoU**, a new best, with `big_rock` up meaningfully
     (0.611 → 0.630) — the class that most needed more decoder capacity to resolve.

## Current best recipe

```
python train_segmentation.py --epochs 50 --aspp-channels 384 --low-level-proj-channels 96
```
(warmup + grad clipping are on by default now; augmentation/Dice stay off unless
explicitly passed).

## Open ideas not yet tried

- `WeightedRandomSampler` to oversample scenes containing `big_rock` pixels
  (data-level imbalance fix, no loss-shape risk).
- Larger `--image-size` for a finer ViT patch grid → sharper rock boundaries (higher
  compute cost per step, encoder still frozen).
- Further decoder widening / an extra fuse conv layer, now that widening helped once.
