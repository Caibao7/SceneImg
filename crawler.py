#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于百度图片搜索抓取室内场景图片的异步爬虫脚本（强化版）。"""

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import random
import re
import time
from io import BytesIO
from typing import Iterable, Optional, Tuple
from urllib.parse import quote_plus

import aiohttp
from PIL import Image, ImageOps

BAIDU_ENDPOINT = "https://image.baidu.com/search/acjson"
BAIDU_SEARCH_PAGE = "https://image.baidu.com/search/index"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)

# 过滤阈值（debug 模式会适当放宽）
MIN_DIMENSION = 400
MAX_DIMENSION = 6000
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5

# 保存格式映射（按内容判断）
FORMAT_EXT = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".jpg",  # 统一转为 jpg 以提高兼容性
    "BMP": ".jpg",
    "GIF": ".jpg",   # 静态首帧保存为 jpg
}

_slug_pattern = re.compile(r"[^\w\-\u4e00-\u9fff]+")


class GlobalDedupIndex:
    """维护跨查询的全局去重索引。"""

    index_filename = "_global_index.jsonl"

    def __init__(self, dest_root: pathlib.Path):
        self.dest_root = dest_root
        self.index_path = dest_root / self.index_filename
        self._hashes: set[str] = set()
        self._url_keys: set[str] = set()
        self._lock = asyncio.Lock()
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sha = record.get("sha1")
                    url_key = record.get("url_md5")
                    if isinstance(sha, str):
                        self._hashes.add(sha)
                    if isinstance(url_key, str):
                        self._url_keys.add(url_key)

    @staticmethod
    def make_url_key(url: str) -> str:
        return hashlib.md5(url.encode("utf-8", errors="ignore")).hexdigest()

    async def claim_url(self, url_key: str) -> bool:
        """尝试占用 URL，若已存在则返回 False。"""
        async with self._lock:
            if url_key in self._url_keys:
                return False
            self._url_keys.add(url_key)
            return True

    async def reserve_hash(self, sha1: str) -> bool:
        """在写盘前占位，避免并发重复写。返回 False 表示重复。"""
        async with self._lock:
            if sha1 in self._hashes:
                return False
            self._hashes.add(sha1)
            return True

    async def rollback_hash(self, sha1: str) -> None:
        async with self._lock:
            self._hashes.discard(sha1)

    async def record(
        self,
        sha1: str,
        url_key: Optional[str],
        url: str,
        relative_path: str,
    ) -> None:
        """记录新的图片条目。"""
        async with self._lock:
            if url_key:
                self._url_keys.add(url_key)
            entry = {
                "sha1": sha1,
                "url_md5": url_key,
                "url": url,
                "path": relative_path,
                "timestamp": time.time(),
            }
            with self.index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

def build_queries(default: bool) -> list[str]:
    base = [
        "室内 场景 日常 杂物",
        "家居 室内 生活 场景 自然 光线",
        "客厅 室内 生活 气息 真实",
        "厨房 室内 桌面 杂物",
        "卧室 室内 稍微 凌乱",
        "书房 室内 桌面 物品",
        "工作室 室内 工具 生活化",
    ]
    return base if default else []


def sanitize(text: str) -> str:
    """保留中文/字母数字/下划线/短横线；为空时回退到 sha1 前缀。"""
    s = _slug_pattern.sub("-", text.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-_")
    if not s:
        s = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]
    return s


def _relative_to_root(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


async def backoff_sleep(base=0.8, factor=2.0, attempt=0, jitter=0.2) -> None:
    """指数退避 + 抖动。"""
    t = base * (factor ** attempt)
    t = t * (1.0 + random.uniform(-jitter, jitter))
    await asyncio.sleep(max(0.1, min(t, 10.0)))


def extract_url(entry: dict) -> Optional[str]:
    """从返回项里尽可能提取可用图片 URL。"""
    candidates = [
        entry.get("hoverURL"),
        entry.get("objURL"),
        entry.get("thumbURL"),
        entry.get("middleURL"),
    ]
    rep = entry.get("replaceUrl")
    if isinstance(rep, list):
        for r in rep:
            if isinstance(r, dict):
                candidates.append(r.get("ObjURL"))
                candidates.append(r.get("FromURL"))

    for u in candidates:
        if isinstance(u, str) and u.startswith(("http://", "https://")):
            return u
    return None


def generate_logid() -> str:
    now_ms = int(time.time() * 1000)
    rand = random.randint(0, 999999)
    return f"{now_ms}{rand:06d}"


async def fetch_page(
    session: aiohttp.ClientSession,
    query: str,
    offset: int,
    retries: int = 3,
    debug: bool = False,
    extra_headers: Optional[dict] = None,
    extra_params: Optional[dict] = None,
) -> dict:
    # gsm 通常是 offset 的 16 进制表示
    gsm = format(offset, "x")
    params = {
        "tn": "resultjson_com",
        "ipn": "rj",
        "ct": "201326592",
        "fp": "result",
        "queryWord": query,
        "word": query,
        "pn": str(offset),
        "rn": "30",
        "ie": "utf-8",
        "oe": "utf-8",
        "nc": "1",     # no constraints
        "lm": "-1",    # 时间不限
        "gsm": gsm,
        "cl": "2",
        "ic": "0",
        "st": "-1",
        "z": "",
        "face": "0",
        "istype": "2",
        "qc": "",
        "fr": "",
        "width": "",
        "height": "",
        "adpicid": "",
        "pn": str(offset),
        "rn": "30",
        "t": str(int(time.time() * 1000)),
    }
    if extra_params:
        params.update(extra_params)
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            async with session.get(BAIDU_ENDPOINT, params=params, headers=extra_headers) as resp:
                # 部分情况下 5xx/429 需要重试
                if resp.status >= 500 or resp.status == 429:
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message="server busy",
                        headers=resp.headers,
                    )
                resp.raise_for_status()
                # 返回内容类型可能不是标准 json；content_type=None 允许宽松解析
                payload = await resp.json(content_type=None)
                if debug:
                    summary = {
                        "keys": list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
                        "length": len(payload.get("data", [])) if isinstance(payload, dict) else None,
                    }
                    print(f"[debug] fetch_page ok offset={offset} summary={summary}")
                    if isinstance(payload, dict) and payload.get("message"):
                        print(f"[debug] fetch_page message={payload.get('message')}")
                return payload
        except Exception as e:
            last_err = e
            if debug:
                print(f"[debug] fetch_page attempt={attempt} offset={offset} err={e}")
            if attempt < retries - 1:
                await backoff_sleep(attempt=attempt)
            else:
                break
    if debug and last_err:
        print(f"[debug] fetch_page failed permanently at offset={offset}: {last_err}")
    return {}


async def warm_up_session(
    session: aiohttp.ClientSession,
    query: str,
    debug: bool = False,
) -> None:
    """Hit the public search page first so Baidu sets cookies before API calls."""
    params = {
        "tn": "baiduimage",
        "word": query,
        "ie": "utf-8",
        "ct": "201326592",
        "lm": "-1",
    }
    try:
        async with session.get(BAIDU_SEARCH_PAGE, params=params) as resp:
            resp.raise_for_status()
            await resp.read()
            if debug:
                print(f"[debug] warm-up success status={resp.status} query={query}")
    except Exception as exc:
        if debug:
            print(f"[debug] warm-up failed query={query} err={exc}")


def should_keep_and_normalize(blob: bytes, debug: bool = False) -> Tuple[bool, bytes, str, dict]:
    """校验尺寸/宽高比并规范保存，返回(保留?, 规范化后的二进制, 后缀, 额外元数据)。"""
    try:
        with Image.open(BytesIO(blob)) as img:
            # 规范方向（EXIF）
            img = ImageOps.exif_transpose(img)
            width, height = img.size

            # # debug 下放宽阈值（暂时禁用尺寸/宽高过滤）
            # min_dim = 300 if debug else MIN_DIMENSION
            # min_aspect = 0.3 if debug else MIN_ASPECT
            # max_aspect = 3.0 if debug else MAX_ASPECT
            #
            # if not (min_dim <= width <= MAX_DIMENSION):
            #     if debug: print(f"[debug] width {width} out of range")
            #     return (False, b"", "", {})
            # if not (min_dim <= height <= MAX_DIMENSION):
            #     if debug: print(f"[debug] height {height} out of range")
            #     return (False, b"", "", {})
            #
            # aspect = width / height if height else 0
            # if not (min_aspect <= aspect <= max_aspect):
            #     if debug: print(f"[debug] aspect {aspect:.3f} out of range")
            #     return (False, b"", "", {})

            fmt = (img.format or "JPEG").upper()
            suffix = FORMAT_EXT.get(fmt, ".jpg")

            # 统一输出通用格式（去 alpha / 调色板）
            save_img = img
            if suffix == ".jpg":
                if img.mode not in ("RGB", "L"):
                    save_img = img.convert("RGB")
                format_name = "JPEG"
                save_args = {"quality": 90}
            else:
                if img.mode in ("P", "RGBA"):
                    save_img = img.convert("RGB")
                format_name = "PNG"
                save_args = {}

            out = BytesIO()
            save_img.save(out, format_name, **save_args)
            meta = {"width": width, "height": height, "format": fmt}
            return (True, out.getvalue(), suffix, meta)
    except Exception as e:
        if debug:
            print(f"[debug] PIL open/normalize failed: {e}")
        return (False, b"", "", {})


async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    dest: pathlib.Path,
    retries: int = 3,
    debug: bool = False,
    referer: Optional[str] = None,
    dedup: Optional[GlobalDedupIndex] = None,
    url_key: Optional[str] = None,
) -> bool:
    ref = "https://image.baidu.com/search/index?tn=baiduimage"
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            headers = session.headers.copy()
            headers["Referer"] = referer or ref
            headers.setdefault("User-Agent", USER_AGENT)
            async with session.get(url, headers=headers) as resp:
                if resp.status in (403, 429) and debug:
                    print(f"[debug] status={resp.status} url={url}")
                if resp.status >= 500 or resp.status in (403, 429):
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info, history=resp.history,
                        status=resp.status, message="fetch blocked", headers=resp.headers
                    )
                resp.raise_for_status()
                blob = await resp.read()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                if debug:
                    print(f"[debug] download retry attempt={attempt} err={e} url={url}")
                await backoff_sleep(attempt=attempt)
                continue
            else:
                if debug:
                    print(f"[debug] download failed permanently: {e} url={url}")
                return False

        ok, normalized, suffix, meta_img = should_keep_and_normalize(blob, debug=debug)
        if not ok:
            return False

        sha1 = hashlib.sha1(normalized).hexdigest()
        reserved = True
        if dedup:
            reserved = await dedup.reserve_hash(sha1)
            if not reserved:
                return False
        out_path = dest / f"{sha1}{suffix}"
        try:
            if out_path.exists():
                if dedup:
                    rel = _relative_to_root(out_path, dedup.dest_root)
                    await dedup.record(sha1, url_key, url, rel)
                return False

            out_path.write_bytes(normalized)
            meta_payload = {
                "source": str(url),
                "sha1": sha1,
                "content_type": resp.headers.get("Content-Type"),
                "status": resp.status,
                "final_url": str(resp.url),
                **meta_img,
            }
            (dest / f"{sha1}.json").write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if dedup:
                rel = _relative_to_root(out_path, dedup.dest_root)
                await dedup.record(sha1, url_key, url, rel)
            return True
        except Exception:
            if dedup and reserved:
                await dedup.rollback_hash(sha1)
            raise

    if debug and last_err:
        print(f"[debug] unreachable url={url} err={last_err}")
    return False


async def crawl_query(
    query: str,
    limit: int,
    dest_root: pathlib.Path,
    per_host: int = 8,
    batch_size: int = 20,
    debug: bool = False,
    dedup: Optional[GlobalDedupIndex] = None,
) -> None:
    slug = sanitize(query)
    out_dir = dest_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit_per_host=per_host, ssl=False)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)

    async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
        tasks: list[asyncio.Task] = []
        seen_urls: set[str] = set()
        downloaded = 0
        offset = 0

        await warm_up_session(session, query, debug=debug)
        encoded_query = quote_plus(query)
        referer_url = f"{BAIDU_SEARCH_PAGE}?tn=baiduimage&word={encoded_query}"
        api_headers = {
            "Referer": referer_url,
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
        }

        while downloaded < limit:
            extra_params = {
                "logid": generate_logid(),
            }
            data = await fetch_page(
                session,
                query,
                offset,
                debug=debug,
                extra_headers=api_headers,
                extra_params=extra_params,
            )
            offset += 30
            entries = [e for e in data.get("data", []) if isinstance(e, dict) and e]
            if not entries:
                if debug:
                    print(f"[debug] no entries at offset={offset-30}")
                break

            batch_urls: list[tuple[str, str]] = []
            for entry in entries:
                url = extract_url(entry)
                if not url:
                    continue
                url_key = GlobalDedupIndex.make_url_key(url)
                if url_key in seen_urls:
                    continue
                seen_urls.add(url_key)
                if dedup:
                    claimed = await dedup.claim_url(url_key)
                    if not claimed:
                        continue
                batch_urls.append((url, url_key))

            if debug:
                print(f"[debug] offset={offset-30}, candidates={len(batch_urls)}")

            for url, url_key in batch_urls:
                tasks.append(
                    asyncio.create_task(
                        download_image(
                            session,
                            url,
                            out_dir,
                            debug=debug,
                            referer=referer_url,
                            dedup=dedup,
                            url_key=url_key,
                        )
                    )
                )
                if len(tasks) >= batch_size:
                    results = await asyncio.gather(*tasks)
                    got = sum(1 for x in results if x)
                    downloaded += got
                    if debug:
                        print(f"[debug] page saved={got}, total={downloaded}")
                    tasks.clear()
                    if downloaded >= limit:
                        break

        if tasks:
            results = await asyncio.gather(*tasks)
            got = sum(1 for x in results if x)
            downloaded += got
            if debug:
                print(f"[debug] tail saved={got}, total={downloaded}")

    print(f"{query} -> {downloaded} images")


def parse_queries(query_arg: Iterable[str], default: bool) -> list[str]:
    queries = list(query_arg)
    if default:
        queries.extend(build_queries(default=True))
    if not queries:
        raise SystemExit("请提供 --query / --query-file，或使用 --default-queries。")
    return queries


def load_query_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


async def run_all(
    queries: list[str],
    limit: int,
    dest_root: pathlib.Path,
    per_host: int,
    batch_size: int,
    debug: bool,
    dedup: Optional[GlobalDedupIndex] = None,
) -> None:
    await asyncio.gather(
        *(crawl_query(
            q,
            limit,
            dest_root,
            per_host=per_host,
            batch_size=batch_size,
            debug=debug,
            dedup=dedup,
        )
          for q in queries)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baidu 室内场景抓取脚本（强化版）")
    parser.add_argument("-o", "--output", default="crawledimg/baidu/raw", help="图片输出目录")
    parser.add_argument("-n", "--limit-per-query", type=int, default=200, help="每个查询抓取的上限数量")
    parser.add_argument("--query", action="append", default=[], help="自定义查询词，可多次提供")
    parser.add_argument("--query-file", help="包含查询词的文件，每行一个，支持注释")
    parser.add_argument("--default-queries", action="store_true", help="使用内置的室内场景查询词")
    parser.add_argument("--debug", action="store_true", help="打印调试信息并放宽过滤阈值")
    parser.add_argument("--per-host", type=int, default=8, help="每主机并发上限（TCPConnector.limit_per_host）")
    parser.add_argument("--batch-size", type=int, default=20, help="攒多少下载任务再 gather 一次")
    args = parser.parse_args()

    queries = list(args.query)
    if args.query_file:
        queries.extend(load_query_file(args.query_file))
    queries = parse_queries(queries, args.default_queries)

    dest_root = pathlib.Path(args.output)
    dest_root.mkdir(parents=True, exist_ok=True)
    dedup_index = GlobalDedupIndex(dest_root)

    asyncio.run(
        run_all(
            queries,
            args.limit_per_query,
            dest_root,
            per_host=args.per_host,
            batch_size=args.batch_size,
            debug=args.debug,
            dedup=dedup_index,
        )
    )


if __name__ == "__main__":
    main()
