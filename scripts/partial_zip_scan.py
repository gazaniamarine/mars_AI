"""Read local file headers sequentially from the start of a zip that is still being
downloaded (append-only, no central directory yet). Safe because each local header
carries its own compressed/uncompressed size when the writer didn't use a streaming
data descriptor (checked via the general-purpose flag's bit 3) -- true for this archive.

Stops a safety margin before the current end of the on-disk file, so it never reads a
truncated entry that's still being written to.
"""

from __future__ import annotations

import os
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

LFH_SIG = b"PK\x03\x04"
SAFETY_MARGIN_BYTES = 64 * 1024 * 1024  # 64MB: comfortably more than one entry, avoids racing the writer


def iter_entries(path: str, safety_margin: int = SAFETY_MARGIN_BYTES):
    boundary = os.path.getsize(path) - safety_margin
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            if pos >= boundary:
                return
            sig = f.read(4)
            if sig != LFH_SIG:
                return  # hit the central directory or end of what's been written cleanly
            hdr = f.read(26)
            if len(hdr) < 26:
                return
            (ver, flag, method, mtime, mdate, crc32, comp_size, uncomp_size,
             fname_len, extra_len) = struct.unpack("<HHHHHIIIHH", hdr)
            if flag & 0x8:
                raise RuntimeError(f"entry at {pos} uses a streaming data descriptor; unsupported")
            fname = f.read(fname_len).decode("utf-8", "replace")
            f.read(extra_len)
            data_start = f.tell()
            if data_start + comp_size > boundary:
                return
            yield dict(name=fname, method=method, comp_size=comp_size,
                       uncomp_size=uncomp_size, data_start=data_start)
            f.seek(comp_size, 1)


def read_entry_bytes(path: str, entry: dict) -> bytes:
    with open(path, "rb") as f:
        f.seek(entry["data_start"])
        raw = f.read(entry["comp_size"])
    if entry["method"] == 8:
        return zlib.decompress(raw, -15)
    if entry["method"] == 0:
        return raw
    raise RuntimeError(f"unsupported compression method {entry['method']} for {entry['name']}")


def main():
    if len(sys.argv) < 2:
        print("usage: partial_zip_scan.py <zip_path> [name_substring_filter]")
        sys.exit(1)
    path = sys.argv[1]
    name_filter = sys.argv[2] if len(sys.argv) > 2 else None

    by_dir = defaultdict(int)
    total = 0
    last_name = None
    for entry in iter_entries(path):
        total += 1
        last_name = entry["name"]
        if name_filter and name_filter not in entry["name"]:
            continue
        parent = str(Path(entry["name"]).parent)
        by_dir[parent] += 1

    print(f"scanned {total} entries from the already-downloaded prefix")
    print(f"last entry seen: {last_name}")
    print(f"\n=== file counts by directory{' (filtered on ' + name_filter + ')' if name_filter else ''} ===")
    for d, count in sorted(by_dir.items(), key=lambda kv: -kv[1])[:40]:
        print(f"{count:8d}  {d}")


if __name__ == "__main__":
    main()
