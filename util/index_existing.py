#!/usr/bin/env python3
"""根据现有图片重建/更新索引文件，支持补全缺失条目以及移除不存在的条目。"""

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Iterable, Optional

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_INDEX_CANDIDATES = ("_filtered_index.jsonl", "_global_index.jsonl")


def detect_index_path(root: pathlib.Path, explicit: Optional[str]) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit).resolve()
    for name in DEFAULT_INDEX_CANDIDATES:
        candidate = root / name
        if candidate.exists():
            return candidate
    root_name = root.name.lower()
    if "filter" in root_name:
        return root / "_filtered_index.jsonl"
    return root / "_global_index.jsonl"


def load_existing_entries(index_path: pathlib.Path) -> list[dict]:
    entries: list[dict] = []
    if not index_path.exists():
        return entries
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def determine_mode(index_path: pathlib.Path, existing: list[dict]) -> str:
    if index_path.name == "_filtered_index.jsonl":
        return "filtered"
    if index_path.name == "_global_index.jsonl":
        return "global"
    for entry in existing:
        if isinstance(entry, dict):
            if "target" in entry:
                return "filtered"
            if "path" in entry:
                return "global"
    return "global"


def iter_images(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for suffix in IMAGE_SUFFIXES:
        yield from root.rglob(f"*{suffix}")


def load_metadata(image_path: pathlib.Path) -> dict:
    meta_path = image_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def sha1_of_bytes(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_sha1(image_path: pathlib.Path, meta: dict) -> str:
    sha1 = meta.get("sha1")
    if isinstance(sha1, str):
        return sha1
    candidate = image_path.stem.lower()
    if len(candidate) == 40 and all(c in "0123456789abcdef" for c in candidate):
        return candidate
    return sha1_of_bytes(image_path)


def build_lookup(existing: list[dict], mode: str) -> tuple[dict[str, dict], dict[str, dict]]:
    by_sha: dict[str, dict] = {}
    by_path: dict[str, dict] = {}
    key_name = "target" if mode == "filtered" else "path"
    for entry in existing:
        if not isinstance(entry, dict):
            continue
        sha = entry.get("sha1")
        if isinstance(sha, str):
            by_sha[sha] = entry
        path_key = entry.get(key_name)
        if isinstance(path_key, str):
            by_path[path_key] = entry
    return by_sha, by_path


def entry_path(entry: dict, mode: str) -> Optional[str]:
    key = "target" if mode == "filtered" else "path"
    value = entry.get(key)
    return value if isinstance(value, str) else None


def build_entry(
    image_path: pathlib.Path,
    root: pathlib.Path,
    mode: str,
    by_sha: dict[str, dict],
    by_path: dict[str, dict],
) -> dict:
    meta = load_metadata(image_path)
    sha1 = choose_sha1(image_path, meta)
    rel_path = image_path
    try:
        rel_path = image_path.relative_to(root)
    except ValueError:
        pass
    rel_str = str(rel_path)

    previous = by_sha.get(sha1) or by_path.get(rel_str)

    timestamp = meta.get("timestamp")
    if not isinstance(timestamp, (int, float)) and previous:
        timestamp = previous.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        timestamp = time.time()

    if mode == "filtered":
        source = meta.get("source")
        if not isinstance(source, str) and previous:
            source = previous.get("source")
        if not isinstance(source, str):
            source = None
        return {
            "sha1": sha1,
            "source": source,
            "target": rel_str,
            "timestamp": timestamp,
        }

    url = meta.get("final_url") or meta.get("source")
    if not isinstance(url, str) and previous:
        url = previous.get("url")
    if not isinstance(url, str):
        url = None
    url_md5 = hashlib.md5(url.encode("utf-8", errors="ignore")).hexdigest() if url else None

    return {
        "sha1": sha1,
        "url_md5": url_md5,
        "url": url,
        "path": rel_str,
        "timestamp": timestamp,
    }


def summarise_changes(existing: list[dict], current: list[dict], mode: str) -> tuple[int, int, int]:
    existing_by_path: dict[str, dict] = {}
    for entry in existing:
        if not isinstance(entry, dict):
            continue
        path_key = entry_path(entry, mode)
        if path_key:
            existing_by_path[path_key] = entry

    current_by_path: dict[str, dict] = {}
    for entry in current:
        if not isinstance(entry, dict):
            continue
        path_key = entry_path(entry, mode)
        if path_key:
            current_by_path[path_key] = entry

    added_keys = set(current_by_path) - set(existing_by_path)
    removed_keys = set(existing_by_path) - set(current_by_path)
    shared_keys = set(current_by_path) & set(existing_by_path)
    updated = sum(
        1
        for key in shared_keys
        if current_by_path[key] != existing_by_path[key]
    )
    return len(added_keys), len(removed_keys), updated


def relative_str(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def rebuild_index(root: pathlib.Path, index_path: pathlib.Path, dry_run: bool) -> None:
    existing_entries = load_existing_entries(index_path)
    mode = determine_mode(index_path, existing_entries)
    by_sha, by_path = build_lookup(existing_entries, mode)

    image_paths = list(iter_images(root))
    image_paths.sort(key=lambda p: relative_str(p, root))

    current_entries = [
        build_entry(image_path, root, mode, by_sha, by_path)
        for image_path in image_paths
    ]

    added, removed, updated = summarise_changes(existing_entries, current_entries, mode)

    print(f"root: {root}")
    print(f"index: {index_path}")
    print(f"mode: {mode}")
    print(f"images: {len(image_paths)}")
    print(f"changes -> added: {added}, removed: {removed}, updated: {updated}")

    if dry_run:
        print("dry-run: 未写入索引文件。")
        return

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as fh:
        for entry in current_entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"已写入 {len(current_entries)} 条记录至 {index_path}。")


def main() -> None:
    parser = argparse.ArgumentParser(description="根据现有图片重建/更新索引文件。")
    parser.add_argument(
        "--root",
        default="crawledimg/baidu/raw",
        help="图片根目录（默认 crawledimg/baidu/raw）",
    )
    parser.add_argument(
        "--index",
        help="显式指定索引文件路径（默认自动推断 _global/_filtered）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出变更统计，不写入索引文件。",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    if not root.exists():
        print(f"指定目录不存在：{root}", file=sys.stderr)
        raise SystemExit(1)

    index_path = detect_index_path(root, args.index)
    rebuild_index(root, index_path, args.dry_run)


if __name__ == "__main__":
    main()
