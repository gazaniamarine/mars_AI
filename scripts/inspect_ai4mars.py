"""Run once ai4mars-dataset-merged-0.6.zip is extracted. Walks the extracted tree,
reports directory structure and file counts, and prints the unique pixel values found
in a sample of label-looking PNGs -- this is what dataset_ai4mars.py's class map must be
built from, instead of assuming a label encoding from memory.

Usage: python scripts/inspect_ai4mars.py /path/to/extracted/ai4mars-dataset-merged-0.6
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def summarize_tree(root: Path, max_depth: int = 3):
    print(f"=== directory structure under {root} (depth<={max_depth}) ===")
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        if path.is_dir():
            n_files = sum(1 for _ in path.iterdir()) if depth == max_depth else None
            suffix = f"  ({n_files} entries)" if n_files is not None else ""
            print("  " * (depth - 1) + rel.name + "/" + suffix)


def sample_label_values(root: Path, keyword: str = "label", sample_size: int = 8):
    candidates = [p for p in root.rglob("*.png") if keyword in str(p).lower()]
    print(f"\n=== found {len(candidates)} PNG files with '{keyword}' in path; sampling {sample_size} ===")
    for p in candidates[:sample_size]:
        arr = np.array(Image.open(p))
        vals, counts = np.unique(arr, return_counts=True)
        print(f"{p.relative_to(root)}: shape={arr.shape} dtype={arr.dtype} "
              f"unique_values={dict(zip(vals.tolist(), counts.tolist()))}")


def sample_image_files(root: Path, exts=(".jpg", ".jpeg", ".png"), sample_size: int = 5):
    counter = Counter()
    example = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            parent = str(p.parent.relative_to(root))
            counter[parent] += 1
            example.setdefault(parent, p)
    print(f"\n=== image/label file counts by parent directory (top 20) ===")
    for parent, count in counter.most_common(20):
        ex = example[parent]
        try:
            with Image.open(ex) as im:
                size = im.size
        except Exception as e:
            size = f"error: {e}"
        print(f"{parent}: {count} files, example={ex.name}, size={size}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"Path does not exist: {root}")
        sys.exit(1)

    summarize_tree(root)
    sample_image_files(root)
    sample_label_values(root)


if __name__ == "__main__":
    main()
