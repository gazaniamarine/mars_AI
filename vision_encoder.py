"""Frozen vision tower extracted from Qwen/Qwen3-VL-4B-Instruct.

Loads only `model.visual.{patch_embed,pos_embed,blocks.*}` from the checkpoint --
`merger` and `deepstack_merger_list` belong to the vision-language merger stage that
this project deliberately stops before, so their weights are never loaded (left at
random init) and their outputs are discarded. The real `Qwen3VLVisionModel.forward`
is reused unmodified (correct patch embedding, rotary position embeddings, and
interpolated absolute position embeddings for dynamic resolutions) -- we only tap two
intermediate layers via forward hooks.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoImageProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

QWEN3_VL_CHECKPOINT = "Qwen/Qwen3-VL-4B-Instruct"

# The checkpoint's own first DeepStack tap (config.vision_config.deepstack_visual_indexes[0]).
# Reused here as a validated "meaningful intermediate layer" choice for the DeepLabV3+
# decoder's low-level skip connection, rather than an arbitrary depth fraction.
LOW_LEVEL_BLOCK_INDEX = 5

_VISUAL_WEIGHT_PREFIXES = (
    "model.visual.patch_embed.",
    "model.visual.pos_embed.",
    "model.visual.blocks.",
)


def _load_visual_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    snapshot_dir = snapshot_download(checkpoint, allow_patterns=["*.safetensors*", "config.json"])
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    wanted_keys = [k for k in weight_map if k.startswith(_VISUAL_WEIGHT_PREFIXES)]
    keys_by_shard: dict[str, list[str]] = {}
    for key in wanted_keys:
        keys_by_shard.setdefault(weight_map[key], []).append(key)

    state_dict = {}
    for shard_file, keys in keys_by_shard.items():
        shard_path = os.path.join(snapshot_dir, shard_file)
        with safe_open(shard_path, framework="pt") as f:
            for key in keys:
                local_key = key[len("model.visual."):]
                state_dict[local_key] = f.get_tensor(key)
    return state_dict


class Qwen3VLVisionEncoder(nn.Module):
    """Frozen SigLIP2-style vision tower; returns low- and high-level spatial feature maps."""

    def __init__(
        self,
        checkpoint: str = QWEN3_VL_CHECKPOINT,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        full_config = Qwen3VLConfig.from_pretrained(checkpoint)
        vision_config = full_config.vision_config

        self.visual = Qwen3VLVisionModel(vision_config)
        missing, unexpected = self.visual.load_state_dict(
            _load_visual_state_dict(checkpoint), strict=False
        )
        # `merger.*` and `deepstack_merger_list.*` are expected to be missing: we never
        # load or use them since they belong to the vision-language merger stage.
        unexpected_non_merger = [k for k in unexpected if "merger" not in k]
        if unexpected_non_merger:
            raise RuntimeError(f"Unexpected checkpoint keys while loading vision tower: {unexpected_non_merger}")
        missing_non_merger = [k for k in missing if "merger" not in k]
        if missing_non_merger:
            raise RuntimeError(f"Missing checkpoint keys while loading vision tower: {missing_non_merger}")

        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)

        self.patch_size = vision_config.patch_size
        self.hidden_size = vision_config.hidden_size
        self.low_level_index = LOW_LEVEL_BLOCK_INDEX
        self.high_level_index = len(self.visual.blocks) - 1

        self.visual.to(device=device, dtype=dtype)
        self.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = dtype

    def preprocess(self, images: list) -> tuple[torch.Tensor, torch.Tensor]:
        """images: list of PIL.Image, all pre-resized by the caller to the same fixed
        resolution (a multiple of `self.patch_size`) so every image yields an identical
        `grid_thw`, letting the caller batch the resulting spatial feature maps."""
        out = self.image_processor(images=images, return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"]

    @torch.no_grad()
    def forward(self, images: list) -> dict[str, torch.Tensor]:
        pixel_values, grid_thw = self.preprocess(images)
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(self.device)

        grids = grid_thw.tolist()
        t0, h0, w0 = grids[0]
        if any((t, h, w) != (t0, h0, w0) for t, h, w in grids):
            raise ValueError(
                "All images in a batch must share the same patch grid (resize to a "
                f"fixed resolution before calling forward); got grids={grids}"
            )
        if t0 != 1:
            raise ValueError(f"Expected static images (temporal grid=1), got t={t0}")

        captured: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx):
            def hook(_module, _inputs, output):
                captured[layer_idx] = output
            return hook

        handles = [
            self.visual.blocks[self.low_level_index].register_forward_hook(make_hook(self.low_level_index)),
            self.visual.blocks[self.high_level_index].register_forward_hook(make_hook(self.high_level_index)),
        ]
        try:
            self.visual(hidden_states=pixel_values, grid_thw=grid_thw)
        finally:
            for h in handles:
                h.remove()

        num_images = len(images)
        low = self._to_spatial(captured[self.low_level_index], num_images, h0, w0)
        high = self._to_spatial(captured[self.high_level_index], num_images, h0, w0)
        return {"low_level": low, "high_level": high}

    def _to_spatial(self, flat_hidden_states: torch.Tensor, batch: int, h: int, w: int) -> torch.Tensor:
        # flat_hidden_states: (batch * h * w, hidden_size), raster order (row-major per image).
        return (
            flat_hidden_states.view(batch, h, w, self.hidden_size)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
