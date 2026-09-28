"""Smoke test: frozen Qwen3-VL-4B vision tower -> DeepLabV3+ decoder, on real pretrained
weights, with dummy images and dummy labels. No AI4Mars data needed. Confirms shapes line
up and gradients flow through the decoder (the encoder stays frozen, by design)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from deeplab_decoder import DeepLabV3PlusDecoder
from vision_encoder import Qwen3VLVisionEncoder

NUM_CLASSES = 5  # placeholder: soil, bedrock, sand, big-rock, null -- confirm against real AI4Mars labels
IMAGE_SIZE = 512  # must be a multiple of patch_size (16)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    print("Loading frozen Qwen3-VL-4B vision tower ...")
    encoder = Qwen3VLVisionEncoder(device=device, dtype=torch.bfloat16)
    print(f"hidden_size={encoder.hidden_size} low_level_index={encoder.low_level_index} "
          f"high_level_index={encoder.high_level_index}")

    decoder = DeepLabV3PlusDecoder(
        num_classes=NUM_CLASSES,
        high_level_channels=encoder.hidden_size,
        low_level_channels=encoder.hidden_size,
    ).to(device)

    batch_size = 2
    rng = np.random.default_rng(0)
    images = [
        Image.fromarray(rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
        for _ in range(batch_size)
    ]
    dummy_labels = torch.randint(0, NUM_CLASSES, (batch_size, IMAGE_SIZE, IMAGE_SIZE), device=device)

    feats = encoder(images)
    low, high = feats["low_level"], feats["high_level"]
    print(f"low_level shape={tuple(low.shape)} high_level shape={tuple(high.shape)}")
    assert low.shape[0] == batch_size and high.shape[0] == batch_size
    assert not low.requires_grad and not high.requires_grad, "encoder output must be detached (frozen)"

    logits = decoder(low.float(), high.float(), output_size=(IMAGE_SIZE, IMAGE_SIZE))
    print(f"logits shape={tuple(logits.shape)}")
    assert logits.shape == (batch_size, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)

    loss = torch.nn.functional.cross_entropy(logits, dummy_labels)
    print(f"loss={loss.item():.4f}")
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in decoder.parameters() if p.grad is not None]
    assert grad_norms, "no gradients reached the decoder"
    assert all(np.isfinite(g) for g in grad_norms), "NaN/Inf gradient in decoder"
    print(f"decoder params with grad: {len(grad_norms)}, mean grad norm: {np.mean(grad_norms):.4f}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
