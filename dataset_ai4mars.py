"""AI4Mars terrain-segmentation dataset, generalized over camera subsets.

Points at the fully extracted dataset root:
    /media/nahar4/2e670039-b303-4e14-b517-49b4319b069d/AI4mars/extracted/ai4mars-dataset-merged-0.6

Every subset's label pixel encoding below was verified against the real extracted
files (scanning label PNGs' actual unique pixel values), not assumed from the
dataset's docs -- its documented "NAV" key (0=soil,1=bedrock,2=sand,3=big rock,
255=NULL) turned out to only match msl_ncam/mer/m2020_nav. msl_mcam was added in a
later dataset version without its own documented key and empirically uses 4, never
255, for NULL:

    msl_mcam   (Mastcam,  9099 pairs): 0=soil 1=bedrock 2=sand 3=big_rock   4=NULL
    msl_ncam   (NavCam,  16064 pairs): 0=soil 1=bedrock 2=sand 3=big_rock 255=NULL
    mer        (Spirit/Opportunity):   0=soil 1=bedrock 2=sand 3=big_rock 255=NULL
    m2020_nav  (Perseverance NavCam):  0=soil 1=bedrock 2=sand 3=big_rock 255=NULL

Deliberately excluded: `m2020/labels/M2020_GEO` (a completely different, much finer
geology taxonomy -- ~20 classes -- not compatible with this 4-class scheme without a
separate remapping decision), and the dataset's `labels/test/*` expert sets (used for
official held-out evaluation, not paired 1:1 with a `train` split the way these are).

Per the dataset's own changelog, data-quality ranking is:
    MSL NavCam > MER > MSL MastCam > M2020 NAV > M2020 Geology

Scene-disjoint splitting
-------------------------
Rover cameras fire in bursts (seconds apart) at one site, so a plain random image-level
train/val split lets near-duplicate frames leak across the split and inflates reported
IoU. Each subset's filename carries either an explicit sol (Mars-day) number or a
spacecraft-clock (SCLK) value; `find_pairs` extracts a `scene_id` per image using
whichever is available (verified against real filenames from each subset), and
`split_pairs` assigns whole scenes to train or val, never splitting one scene across
both:
    msl_mcam   -- sol number is the filename's first digit run (e.g. "1150...")
    m2020_nav  -- sol number is the filename's second digit run (explicit `_0002_` field)
    msl_ncam   -- no sol field in the filename; cluster by SCLK (first digit run) with a
                  new scene starting whenever the gap to the previous frame exceeds 1
                  hour -- empirically, gaps within one imaging burst are almost always
                  under ~5 minutes (75th percentile 298s) while between-session gaps jump
                  to hours/days (90th percentile 7054s), so 1 hour cleanly separates them
    mer        -- same SCLK-gap clustering as msl_ncam (its SCLK is the second digit run)
"""

from __future__ import annotations

import glob
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
NUM_CLASSES = len(CLASS_NAMES)

# Canonical NULL marker after loading -- every subset's raw ignore value gets remapped
# to this in AI4MarsSegDataset, so a combined batch from multiple subsets (which use
# different raw NULL markers) can still share one ignore_index in the loss/metrics.
IGNORE_INDEX = 255

_DIGIT_RUN_RE = re.compile(r"\d+")

def _mcam_label_stem(label_filename: str) -> str:
    # "<stem>_<id>_merged.png" -- strip the last two underscore-separated segments
    return label_filename.rsplit("_", 2)[0]


def _exact_label_stem(label_filename: str) -> str:
    # "<stem>.png"
    return label_filename[:-4]


_MERGED_SUFFIX_RE = re.compile(r"_merged\d+\.png$")


def _merged_suffix_label_stem(label_filename: str) -> str:
    # "<stem>_merged<digits>.png"
    return _MERGED_SUFFIX_RE.sub("", label_filename)


CAMERA_SUBSETS = {
    "msl_mcam": dict(
        images_dir="msl/mcam/images",
        image_exts=[".JPG"],
        labels_dir="msl/mcam/labels/train",
        label_stem_fn=_mcam_label_stem,
        raw_ignore_index=4,
        scene_key=dict(kind="direct", digit_index=0),  # sol
    ),
    "msl_ncam": dict(
        images_dir="msl/ncam/images/edr",
        image_exts=[".JPG"],
        labels_dir="msl/ncam/labels/train",
        label_stem_fn=_exact_label_stem,
        raw_ignore_index=255,
        scene_key=dict(kind="sclk_cluster", digit_index=0, gap_threshold=3600),
    ),
    "mer": dict(
        images_dir="mer/images/eff",
        image_exts=[".JPG"],
        labels_dir="mer/labels/train/merged-unmasked",
        label_stem_fn=_merged_suffix_label_stem,
        raw_ignore_index=255,
        scene_key=dict(kind="sclk_cluster", digit_index=1, gap_threshold=3600),
    ),
    "m2020_nav": dict(
        images_dir="m2020/images/ncam",
        image_exts=[".jpeg", ".jpg"],  # this directory mixes both extensions
        labels_dir="m2020/labels/NAV",
        label_stem_fn=_merged_suffix_label_stem,
        raw_ignore_index=255,
        scene_key=dict(kind="direct", digit_index=1),  # sol
    ),
}


def _build_label_lookup(lbl_dir: str, label_stem_fn) -> dict[str, list[str]]:
    lookup = defaultdict(list)
    for fname in os.listdir(lbl_dir):
        if not fname.endswith(".png"):
            continue
        lookup[label_stem_fn(fname)].append(os.path.join(lbl_dir, fname))
    return lookup


def _assign_scene_ids(subset: str, img_paths: list[str], cfg: dict) -> list[str]:
    digit_index = cfg["scene_key"]["digit_index"]
    keys = [int(_DIGIT_RUN_RE.findall(os.path.basename(p))[digit_index]) for p in img_paths]

    if cfg["scene_key"]["kind"] == "direct":
        return [f"{subset}_sol{k}" for k in keys]

    # sclk_cluster: sort by clock value, start a new scene whenever the gap to the
    # previous frame exceeds gap_threshold seconds.
    gap_threshold = cfg["scene_key"]["gap_threshold"]
    order = sorted(range(len(img_paths)), key=lambda i: keys[i])
    scene_ids = [None] * len(img_paths)
    cluster_id, prev_key = 0, None
    for i in order:
        if prev_key is not None and keys[i] - prev_key > gap_threshold:
            cluster_id += 1
        scene_ids[i] = f"{subset}_scene{cluster_id}"
        prev_key = keys[i]
    return scene_ids


def find_pairs(dataset_root: str, subset: str) -> list[tuple[str, str, int, str]]:
    """Returns (image_path, label_path, raw_ignore_index, scene_id) tuples for one subset."""
    cfg = CAMERA_SUBSETS[subset]
    img_dir = os.path.join(dataset_root, cfg["images_dir"])
    lbl_dir = os.path.join(dataset_root, cfg["labels_dir"])

    img_paths = []
    for ext in cfg["image_exts"]:
        img_paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    img_paths.sort()
    scene_ids = _assign_scene_ids(subset, img_paths, cfg)

    label_lookup = _build_label_lookup(lbl_dir, cfg["label_stem_fn"])

    pairs = []
    for img_path, scene_id in zip(img_paths, scene_ids):
        base = os.path.basename(img_path)
        ext = next(e for e in cfg["image_exts"] if base.endswith(e))
        stem = base[: -len(ext)]
        candidates = label_lookup.get(stem, [])
        if len(candidates) == 0:
            continue  # not every image has a train label meeting the agreement threshold
        if len(candidates) > 1:
            raise ValueError(f"ambiguous label match for {stem} in {subset}: {candidates}")
        pairs.append((img_path, candidates[0], cfg["raw_ignore_index"], scene_id))
    return pairs


def find_all_pairs(dataset_root: str, subsets: list[str] | None = None) -> list[tuple[str, str, int, str]]:
    subsets = subsets if subsets is not None else list(CAMERA_SUBSETS)
    pairs = []
    for subset in subsets:
        pairs.extend(find_pairs(dataset_root, subset))
    return pairs


class AI4MarsSegDataset(Dataset):
    """`augment=True` (train split) applies a synchronized random-resized-crop (scale
    jitter 0.75-1.25x + crop back to image_size) and random horizontal flip to the
    image+label pair, plus color jitter to the image only. `augment=False` (val split)
    just resizes, matching the original deterministic behavior."""

    def __init__(self, pairs: list[tuple[str, str, int, str]], image_size: int = 512, augment: bool = False):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        if augment:
            self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image: Image.Image, label: Image.Image) -> tuple[Image.Image, Image.Image]:
        w, h = image.size
        scale = random.uniform(0.75, 1.25)
        crop_size = max(1, round(self.image_size / scale))

        if w < crop_size or h < crop_size:
            up = max(crop_size / w, crop_size / h)
            new_w, new_h = max(crop_size, round(w * up)), max(crop_size, round(h * up))
            image = image.resize((new_w, new_h), Image.BILINEAR)
            label = label.resize((new_w, new_h), Image.NEAREST)
            w, h = new_w, new_h

        x0 = random.randint(0, w - crop_size)
        y0 = random.randint(0, h - crop_size)
        box = (x0, y0, x0 + crop_size, y0 + crop_size)
        image = image.crop(box).resize((self.image_size, self.image_size), Image.BILINEAR)
        label = label.crop(box).resize((self.image_size, self.image_size), Image.NEAREST)

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        image = self.color_jitter(image)
        return image, label

    def __getitem__(self, idx: int):
        img_path, lbl_path, raw_ignore_index, _scene_id = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path)

        if self.augment:
            image, label = self._augment(image, label)
        else:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            label = label.resize((self.image_size, self.image_size), Image.NEAREST)

        label_arr = np.array(label, dtype=np.int64)
        if raw_ignore_index != IGNORE_INDEX:
            label_arr[label_arr == raw_ignore_index] = IGNORE_INDEX
        return image, torch.from_numpy(label_arr)


def split_pairs(pairs: list[tuple[str, str, int, str]], val_fraction: float = 0.1, seed: int = 0):
    """Scene-disjoint split: whole scenes (bursts of near-duplicate frames) go entirely
    to train or entirely to val, never both -- see module docstring."""
    by_scene = defaultdict(list)
    for p in pairs:
        by_scene[p[3]].append(p)

    scene_ids = list(by_scene)
    random.Random(seed).shuffle(scene_ids)

    target_val = int(len(pairs) * val_fraction)
    val_pairs, val_count = [], 0
    val_scene_ids = set()
    for sid in scene_ids:
        if val_count >= target_val:
            break
        val_scene_ids.add(sid)
        val_count += len(by_scene[sid])

    train_pairs, val_pairs = [], []
    for p in pairs:
        (val_pairs if p[3] in val_scene_ids else train_pairs).append(p)
    return train_pairs, val_pairs


def collate_images_labels(batch):
    images, labels = zip(*batch)
    return list(images), torch.stack(labels, dim=0)
