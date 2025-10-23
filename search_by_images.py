#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用百度以图搜图接口，根据种子图片批量抓取相似室内场景图片（2025 版：graph.s HTML 解析 + 真分页 + 相似度筛选）"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import pathlib
import random
import re
import time
import urllib.parse
from io import BytesIO
from typing import Iterable, Optional, Tuple, Dict, Any, List, Set

import aiohttp
from PIL import Image, ImageOps

import imghdr
from urllib.parse import urlparse

import html as html_lib
from urllib.parse import urljoin, quote_plus

_SIMILAR_URL_PATTERNS = [
    r'(?:src|href|data-url)\s*=\s*["\']((?:https?:)?//graph\.baidu\.com/pcpage/similar[^"\']+)["\']',
    r'["\'](/pcpage/similar[^"\']+)["\']',
    r'(https?://graph\.baidu\.com/pcpage/similar[^\s"<>\']+)',
]

_SIMILAR_AJAX_RE = re.compile(
    r'(https://graph\.baidu\.com/ajax/similardetailnew\?[^"\'<>]+)'
)

import html as html_lib

import re, html as html_lib, base64, json

def _decode_unicode_url(url: str) -> str:
    try:
        return url.encode("utf-8").decode("unicode_escape")
    except Exception:
        return url


def _extract_ajax_params(text: str) -> dict[str, str]:
    for match in _SIMILAR_AJAX_RE.finditer(text):
        raw = html_lib.unescape(match.group(1))
        raw = _decode_unicode_url(raw)
        try:
            parsed = urllib.parse.urlparse(raw)
            query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        except Exception:
            continue
        params = {k: v for k, v in query}
        if params:
            return params
    return {}


def _extract_similar_params(html_main: str) -> dict:
    """
    从 graph.baidu.com/s 主页面中提取 queryImageUrl、token、carousel。
    支持新版 window.__PRELOADED_STATE__ Base64 格式。
    """
    text = html_lib.unescape(html_main)

    # 1️⃣ 提取图片 URL
    image = None
    for pat in [
        r'"queryImageUrl"\s*:\s*"(https?://mms\d+\.baidu\.com/[^"]+)"',
        r'"image"\s*:\s*"(https?://mms\d+\.baidu\.com/[^"]+)"',
        r'(https?://mms\d+\.baidu\.com/[^\s"\'<>]+)',
        r'(https?://img\d+\.baidu\.com/[^\s"\'<>]+)',
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            image = html_lib.unescape(m.group(m.lastindex or 1))
            break

    if image:
        image = image.strip()
        image = re.split(r'\);|["\'<>]', image)[0].strip()

    # 2️⃣ 提取 token（新版百度页面多为 window.__PRELOADED_STATE__ = "base64(...)"）
    token = None
    # 尝试直接匹配普通形式
    for pat in [
        r'"token"\s*:\s*"([0-9a-z]{16,64})"',
        r'window.__INITIAL_STATE__[^{}]+token["\']\s*:\s*["\']([0-9a-z]{16,64})',
        r'token\s*=\s*["\']([0-9a-z]{16,64})["\']',
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            token = m.group(1)
            break

    # 尝试新版 Base64 格式
    if not token:
        m = re.search(r'__PRELOADED_STATE__\s*=\s*"([^"]+)"', text)
        if m:
            try:
                decoded = base64.b64decode(m.group(1)).decode("utf-8", "ignore")
                j = json.loads(decoded)
                if isinstance(j, dict):
                    # 新版结构中 token 多在 ["similarResult"]["token"] 下
                    token = (
                        j.get("similarResult", {}).get("token")
                        or j.get("similarList", {}).get("token")
                        or j.get("token")
                    )
            except Exception:
                pass

    # 3️⃣ carousel
    carousel = None
    m = re.search(r'carousel\s*[:=]\s*["\']?(\d{3,6})["\']?', text)
    if m:
        carousel = m.group(1)

    ajax_params = _extract_ajax_params(text)

    return {"image": image, "token": token, "carousel": carousel, "ajax_params": ajax_params}


def _extract_similar_iframe(main_html: str) -> Optional[str]:
    """尽量从主页 HTML 中提取 /pcpage/similar 的完整 URL（自动反转义 &amp;、\\u0026）。"""
    text = html_lib.unescape(main_html)  # &amp; -> &
    for pat in _SIMILAR_URL_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            u = m.group(1)
            if u.startswith("//"):
                u = "https:" + u
            return u
    # 有些把 URL 放 JSON 里，先搜关键字，再向两侧扩展
    idx = text.find("pcpage/similar")
    if idx != -1:
        l = idx
        r = idx
        L = len(text)
        while l > 0 and text[l] not in "\"'<> \n\r\t":
            l -= 1
        while r < L and text[r] not in "\"'<> \n\r\t":
            r += 1
        u = text[l+1:r]
        u = html_lib.unescape(u)
        if u.startswith("//"):
            u = "https:" + u
        if u.startswith("/"):
            u = urljoin("https://graph.baidu.com/", u)
        if "pcpage/similar" in u:
            return u
    return None


def _extract_token(main_html: str) -> Optional[str]:
    """从主页里尽力抽取 shtiuToken（可缺省）。"""
    text = html_lib.unescape(main_html)
    m = re.search(r'shtiuToken\s*[:=]\s*["\']([0-9a-fA-F]+)["\']', text)
    if m:
        return m.group(1)
    return None


# 放在文件顶部 imports 附近
MIME_EXT_MAP = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/avif": ".avif",
    "image/tiff": ".tif",
    "image/x-icon": ".ico",
    "image/svg+xml": ".svg",
}

def _guess_ext_from_url(url: str) -> str:
    try:
        path = urlparse(url).path.lower()
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".ico", ".svg"):
            if path.endswith(ext):
                return ext
    except Exception:
        pass
    return ""

def _guess_ext_from_magic(blob: bytes) -> str:
    try:
        kind = imghdr.what(None, h=blob)  # 可能返回 'jpeg' 'png' 'gif' 'bmp' 'webp' 'tiff' 等
        if kind == "jpeg":
            return ".jpg"
        if kind == "tiff":
            return ".tif"
        if kind:
            return "." + kind
    except Exception:
        pass
    return ""


GRAPH_UPLOAD_ENDPOINT = "https://graph.baidu.com/upload"
GRAPH_SEARCH_PAGE = "https://graph.baidu.com/s"
BAIDU_VIEW_ENDPOINT = "https://image.baidu.com/pcdutu"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)

# 基本图片过滤
MIN_DIMENSION = 400
MAX_DIMENSION = 6000
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5

# dHash 相关度阈值（越小越相似），建议 12~22 之间
DEFAULT_DHASH_THRESHOLD = 18
DEFAULT_DHASH_MIN_DISTANCE = 4

FORMAT_EXT = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".jpg",
    "BMP": ".jpg",
    "GIF": ".jpg",
}

_slug_pattern = re.compile(r"[^\w\-\u4e00-\u9fff]+")

# ======================= 工具与去重 =======================

class GlobalDedupIndex:
    """维护跨搜索的全局去重索引。"""
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
                    line = line.strip()
                    if not line:
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
        async with self._lock:
            if url_key in self._url_keys:
                return False
            self._url_keys.add(url_key)
            return True

    async def reserve_hash(self, sha1: str) -> bool:
        async with self._lock:
            if sha1 in self._hashes:
                return False
            self._hashes.add(sha1)
            return True

    async def rollback_hash(self, sha1: str) -> None:
        async with self._lock:
            self._hashes.discard(sha1)

    async def record(self, sha1: str, url_key: Optional[str], url: str, relative_path: str) -> None:
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


def sanitize(text: str) -> str:
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
    t = base * (factor ** attempt)
    t = t * (1.0 + random.uniform(-jitter, jitter))
    await asyncio.sleep(max(0.1, min(t, 8.0)))


# ======================= 图片规范化 & dHash =======================

def should_keep_and_normalize(blob: bytes, debug: bool = False) -> tuple[bool, bytes, str, dict]:
    try:
        with Image.open(BytesIO(blob)) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size

            if not debug:
                if not (MIN_DIMENSION <= width <= MAX_DIMENSION):
                    return (False, b"", "", {})
                if not (MIN_DIMENSION <= height <= MAX_DIMENSION):
                    return (False, b"", "", {})
                aspect = width / height if height else 0
                if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                    return (False, b"", "", {})

            fmt = (img.format or "JPEG").upper()
            suffix = FORMAT_EXT.get(fmt, ".jpg")
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
    except Exception as exc:
        if debug:
            print(f"[debug] normalize failed: {exc}")
        return (False, b"", "", {})


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """简易 dHash（差值哈希）"""
    img = ImageOps.exif_transpose(img).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    diff = []
    for y in range(hash_size):
        for x in range(hash_size):
            diff.append(img.getpixel((x, y)) > img.getpixel((x + 1, y)))
    value = 0
    for i, v in enumerate(diff):
        if v:
            value |= 1 << i
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    dest: pathlib.Path,
    *,
    dedup: Optional[GlobalDedupIndex],
    url_key: Optional[str],
    referer: str,
    debug: bool = False,
) -> tuple[bool, Optional[bytes], Optional[str]]:
    ref = referer or "https://image.baidu.com/"
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            headers = session.headers.copy()
            headers["Referer"] = ref
            headers.setdefault("User-Agent", USER_AGENT)
            async with session.get(url, headers=headers, allow_redirects=True) as resp:
                if resp.status >= 500 or resp.status in (403, 429):
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message="fetch blocked",
                        headers=resp.headers,
                    )
                resp.raise_for_status()
                blob = await resp.read()
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                if debug:
                    print(f"[debug] download retry attempt={attempt} url={url} err={exc}")
                await backoff_sleep(attempt=attempt)
                continue
            if debug:
                print(f"[debug] download failed url={url} err={exc}")
            return (False, None, None)

        ok, normalized, suffix, meta_img = should_keep_and_normalize(blob, debug=debug)
        if not ok:
            if debug:
                print(f"[debug] normalize reject url={url}")
            return (False, None, None)

        sha1 = hashlib.sha1(normalized).hexdigest()
        reserved = True
        if dedup:
            reserved = await dedup.reserve_hash(sha1)
            if not reserved:
                return (False, None, None)
        out_path = dest / f"{sha1}{suffix}"
        try:
            if out_path.exists():
                if dedup:
                    rel = _relative_to_root(out_path, dedup.dest_root)
                    await dedup.record(sha1, url_key, url, rel)
                return (False, None, None)

            out_path.write_bytes(normalized)
            meta_payload = {
                "source": str(url),
                "sha1": sha1,
                **meta_img,
            }
            (dest / f"{sha1}.json").write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if dedup:
                rel = _relative_to_root(out_path, dedup.dest_root)
                await dedup.record(sha1, url_key, url, rel)
            return (True, normalized, suffix)
        except Exception:
            if dedup and reserved:
                await dedup.rollback_hash(sha1)
            raise

    if debug and last_err:
        print(f"[debug] unreachable url={url} err={last_err}")
    return (False, None, None)

# ======================= 上传 & 解析 =======================

def build_graph_page(sign: str, query_image_url: str, *, pn: int = 0, rn: int = 60) -> str:
    # graph 页面支持分页参数（经验值）：pn 偏移、rn 数量（30/60）
    qs = {
        "sign": sign,
        "from": "pc",
        "queryImageUrl": query_image_url,
        "pn": str(pn),
        "rn": str(rn),
        "pageFrom": "graph_upload_wise",
    }
    return f"{GRAPH_SEARCH_PAGE}?{urllib.parse.urlencode(qs)}"


async def resolve_query_image_url(session: aiohttp.ClientSession, page_url: str, *, debug: bool = False) -> Optional[str]:
    try:
        async with session.get(page_url, headers={"Referer": "https://image.baidu.com/"}) as resp:
            resp.raise_for_status()
            html = await resp.text()
    except Exception as exc:
        if debug:
            print(f"[debug] resolve queryImageUrl failed: {exc}")
        return None
    m = re.search(r'"queryImageUrl"\s*:\s*"([^"]+)"', html) or re.search(r'queryImageUrl\s*=\s*"([^"]+)"', html)
    if not m:
        m = re.search(r'data-imgurl="([^"]+)"', html) or re.search(r'"imgUrl"\s*:\s*"([^"]+)"', html)
    if not m:
        if debug:
            print("[debug] queryImageUrl not found in HTML")
        return None
    qurl = m.group(1)
    if qurl.startswith("//"):
        qurl = "https:" + qurl
    if debug:
        print(f"[debug] extracted queryImageUrl = {qurl}")
    return qurl


async def prepare_similar_context(
    session: aiohttp.ClientSession,
    main_url: str,
    *,
    debug: bool = False,
) -> dict:
    try:
        async with session.get(main_url, headers={"User-Agent": USER_AGENT, "Referer": "https://image.baidu.com/"}) as resp:
            resp.raise_for_status()
            html_main = await resp.text()
            if debug:
                print(f"[debug] main HTML len={len(html_main)} from {main_url}")
    except Exception as exc:
        if debug:
            print(f"[debug] failed to fetch main graph page: {exc}")
        return {}
    params = _extract_similar_params(html_main)
    params["main_url"] = main_url
    try:
        parsed = urllib.parse.urlsplit(main_url)
        qs = dict(urllib.parse.parse_qsl(parsed.query))
        if "sign" in qs:
            params.setdefault("sign", qs.get("sign"))
    except Exception:
        pass
    return params


def _decode_objurl(url: str) -> str:
    table = {"_z2C$q": ":", "_z&e3B": ".", "AzdH3F": "/"}
    try:
        for k, v in table.items():
            url = url.replace(k, v)
        trans = str.maketrans("wkv1ju2it3hs4g5rq6fp7eo8dn9cm0bla", "abcdefghijklmnopqrstuvwxyz0123456")
        return url.translate(trans)
    except Exception:
        return url


def _extract_entry_image(entry: dict[str, Any]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None

    candidates: list[str] = []

    obj_url = entry.get("objUrl")
    if isinstance(obj_url, str) and obj_url:
        try:
            parsed = urllib.parse.urlparse(obj_url)
            qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            imgs = qs.get("image") or qs.get("imgurl") or qs.get("imgUrl")
            if imgs:
                candidates.extend(imgs)
        except Exception:
            pass

    thumb_url = entry.get("thumbUrl") or entry.get("middleUrl")
    if isinstance(thumb_url, str) and thumb_url:
        candidates.append(thumb_url)

    for url in candidates:
        if not isinstance(url, str):
            continue
        u = html_lib.unescape(url.strip())
        if not u:
            continue
        if u.startswith("//"):
            u = "https:" + u
        if u.startswith("http"):
            return u
    return None


# ------- 上传：同你之前 v2 变体为主，其它作为兜底 -------
async def _upload_v2(session: aiohttp.ClientSession, image_path: pathlib.Path, *, debug: bool=False) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    raw = image_path.read_bytes()
    mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    form = aiohttp.FormData()
    form.add_field("image", raw, filename=image_path.name, content_type=mime)
    headers = {
        "Referer": "https://image.baidu.com/",
        "Origin": "https://image.baidu.com",
    }
    async with session.post(GRAPH_UPLOAD_ENDPOINT, data=form, headers=headers) as resp:
        payload = await resp.json(content_type=None)
    if debug:
        print(f"[debug] upload resp (v2) seed={image_path.name} payload={payload}")
    if not isinstance(payload, dict):
        return None, None, None
    data = payload.get("data") or {}
    page = data.get("url")
    sign = data.get("sign") or data.get("querySign") or data.get("uqid")
    return None, sign, page  # queryImageUrl 可稍后解析


async def upload_reference_image(session: aiohttp.ClientSession, image_path: pathlib.Path, *, debug: bool=False) -> tuple[str, str, Optional[str]]:
    qurl, sign, page = await _upload_v2(session, image_path, debug=debug)
    if not sign and page:
        # 仍尝试从页面提取 sign
        qs = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(page).query))
        sign = sign or qs.get("sign", "")
    if not page and sign:
        page = build_graph_page(sign, "", pn=0, rn=60)
    return qurl or "", sign or "", page


async def warm_up_search(session: aiohttp.ClientSession, query_image_url: str, query_sign: str, page_url: Optional[str], *, debug: bool=False) -> None:
    targets = []
    if page_url:
        targets.append(page_url)
    if query_image_url and query_sign:
        targets.append(build_graph_page(query_sign, query_image_url, pn=0, rn=60))
    for target in targets:
        try:
            async with session.get(target, headers={"Referer": "https://image.baidu.com/"}) as resp:
                resp.raise_for_status()
                await resp.read()
                if debug:
                    print(f"[debug] warm-up ok status={resp.status} url={target}")
                break
        except Exception as exc:
            if debug:
                print(f"[debug] warm-up failed url={target} err={exc}")
            continue


# ======================= HTML 解析 & 真分页 =======================

IMG_URL_RE = re.compile(
    r'(?:data-imgurl|data-src|src)="(https?://[^"]+?)"',
    flags=re.IGNORECASE,
)

HREF_IMGURL_RE = re.compile(
    r'href="(?:/search/detail|/s\?)?[^"]*?(?:imgUrl|imgurl|img)=([^"&]+)',
    flags=re.IGNORECASE,
)

INLINE_JSON_BLOCK_RE = re.compile(
    r'(\{[^<>]{200,}\})',  # 捕获较大的 JSON 块，后续尝试解析
    flags=re.DOTALL,
)

URL_IN_JSON_RE = re.compile(
    r'"(hoverURL|middleURL|thumbURL|objURL)"\s*:\s*"([^"]+)"',
    flags=re.IGNORECASE,
)

def _extract_from_html(html: str) -> Set[str]:
    urls: Set[str] = set()

    # 1) 直接的 img/data-src/data-imgurl
    for m in IMG_URL_RE.finditer(html):
        u = m.group(1)
        if u.startswith("//"):
            u = "https:" + u
        urls.add(u)

    # 2) 详情链接里带的 imgUrl 参数
    for m in HREF_IMGURL_RE.finditer(html):
        enc = m.group(1)
        try:
            u = urllib.parse.unquote(enc)
            if u.startswith("//"):
                u = "https:" + u
            if u.startswith("http"):
                urls.add(u)
        except Exception:
            pass

    # 3) 内联 JSON 里可能直接给了一批 URL
    #    为避免完整 JSON 解析失败，这里只做键值对级别的正则提取
    for jm in URL_IN_JSON_RE.finditer(html):
        u = jm.group(2)
        if u.startswith("//"):
            u = "https:" + u
        # objURL 要尝试解码
        if jm.group(1).lower() == "objurl":
            u_dec = _decode_objurl(u)
            if u_dec.startswith("http"):
                urls.add(u_dec)
        if u.startswith("http"):
            urls.add(u)

    return urls


import re
from urllib.parse import urljoin

async def fetch_similar_page(
    session: aiohttp.ClientSession,
    context: dict[str, Any],
    page_index: int,
    *,
    referer: str,
    debug: bool = False,
) -> dict:
    ajax_params = context.get("ajax_params") if isinstance(context, dict) else None
    if ajax_params:
        query = dict(ajax_params)
        size_raw = query.get("page_size") or query.get("pagesize")
        try:
            page_size = int(size_raw) if size_raw else 30
        except ValueError:
            page_size = 30
        page_size = max(page_size, 30)
        page_size = min(page_size, 60)
        page_num = max(1, page_index + 1)
        query["page"] = str(page_num)
        query["page_size"] = str(page_size)
        if "next" in query:
            query["next"] = str(page_num + 1)
        url = "https://graph.baidu.com/ajax/similardetailnew?" + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)
        headers = {
            "User-Agent": USER_AGENT,
            "Referer": referer,
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
        }
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status >= 500 or resp.status in (403, 429):
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message="ajax blocked",
                        headers=resp.headers,
                    )
                resp.raise_for_status()
                payload = await resp.json(content_type=None)
        except Exception as exc:
            if debug:
                print(f"[debug] ajax similar fetch failed page={page_num} err={exc}")
            return {}

        entries = payload.get("data", {}).get("list", []) if isinstance(payload, dict) else []
        normalized: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            type_info = entry.get("typeInfo")
            if isinstance(type_info, dict):
                entry_type = str(type_info.get("type") or "").lower()
                if entry_type and entry_type not in ("image", "pic", "picture"):
                    continue
            if entry.get("isVideo"):
                continue
            image_url = _extract_entry_image(entry)
            if not image_url:
                continue
            norm = {
                "middleURL": image_url,
            }
            obj = entry.get("objUrl") or entry.get("objURL")
            if isinstance(obj, str):
                norm["objURL"] = obj
            thumb = entry.get("thumbUrl") or entry.get("thumbURL")
            if isinstance(thumb, str):
                norm["thumbURL"] = thumb
            normalized.append(norm)

        if debug:
            print(f"[debug] ajax similar page={page_num} entries={len(normalized)}")

        return {"data": normalized}

    image_raw = context.get("image") if isinstance(context, dict) else None
    token = context.get("token") if isinstance(context, dict) else None
    carousel = context.get("carousel") if isinstance(context, dict) else None
    carousel = carousel or "5038"

    if debug:
        print(f"[debug] fallback similar params image={bool(image_raw)} token={token} carousel={carousel}")

    if not image_raw or not token:
        if debug:
            print("[debug] missing image/token and no ajax params; abort this seed")
        return {}

    qs = {
        "entrance": "GENERAL",
        "tpl_from": "pc",
        "sign": context.get("sign", ""),
        "carousel": carousel,
        "token": token,
        "queryImageUrl": image_raw,
        "pn": str(page_index * 30),
        "rn": "60",
    }
    iframe_url = "https://graph.baidu.com/pcpage/similar?" + urllib.parse.urlencode(qs, quote_via=urllib.parse.quote)

    main_url = context.get("main_url") if isinstance(context, dict) else referer
    if debug:
        print(f"[debug] similar iframe url: {iframe_url}")

    try:
        async with session.get(iframe_url, headers={"User-Agent": USER_AGENT, "Referer": main_url}) as resp:
            if resp.status == 400 and debug:
                body = await resp.text()
                print(f"[debug] similar 400 body head: {body[:200]}")
            resp.raise_for_status()
            html_sim = await resp.text()
            if debug:
                print(f"[debug] iframe HTML len={len(html_sim)} from {iframe_url}")
    except Exception as exc:
        if debug:
            print(f"[debug] failed to fetch iframe: {exc}")
        return {}

    urls: set[str] = set()
    for m in re.finditer(r'data-imgurl="(https?://[^"]+)"', html_sim, flags=re.IGNORECASE):
        urls.add(html_lib.unescape(m.group(1)))
    for m in re.finditer(r'(?:data-src|src)="(https?://[^"]+)"', html_sim, flags=re.IGNORECASE):
        u = html_lib.unescape(m.group(1))
        if u.startswith("http"):
            urls.add(u)

    if debug:
        print(f"[debug] parsed {len(urls)} similar image URLs (fallback)")

    return {"data": [{"middleURL": u} for u in urls]}



# ======================= 主流程 =======================

def gather_seed_images(root: pathlib.Path, glob: Optional[str]) -> list[pathlib.Path]:
    if not root.exists():
        raise FileNotFoundError(f"输入目录不存在：{root}")
    paths = root.glob(glob) if glob else root.rglob("*")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in exts)


def create_output_dir(dest_root: pathlib.Path, slug: str, use_timestamp_subdir: bool) -> pathlib.Path:
    base_dir = dest_root / slug
    if not use_timestamp_subdir:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = base_dir / timestamp
    counter = 1
    while out_dir.exists():
        out_dir = base_dir / f"{timestamp}-{counter:02d}"
        counter += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


async def crawl_reference_image(
    session: aiohttp.ClientSession,
    image_path: pathlib.Path,
    *,
    dest_root: pathlib.Path,
    limit: int,
    batch_size: int,
    dedup: Optional[GlobalDedupIndex],
    use_timestamp_subdir: bool,
    phash_filter: bool,
    dhash_threshold: int,
    dhash_min_distance: int,
    debug: bool = False,
) -> None:
    slug = sanitize(image_path.stem)
    out_dir = create_output_dir(dest_root, slug, use_timestamp_subdir)
    try:
        query_image_url, query_sign, page_url = await upload_reference_image(session, image_path, debug=debug)
    except Exception as exc:
        print(f"[error] 上传失败 {image_path}: {exc}")
        return

    await warm_up_search(session, query_image_url, query_sign, page_url, debug=debug)

    # 补齐 query_image_url
    if (not query_image_url) and page_url:
        fixed = await resolve_query_image_url(session, page_url, debug=debug)
        if fixed:
            query_image_url = fixed

    referer = page_url or build_graph_page(query_sign, query_image_url, pn=0, rn=60)

    # 计算 seed 的 dHash
    seed_hash: Optional[int] = None
    if phash_filter:
        try:
            with Image.open(image_path) as im:
                seed_hash = dhash(im)
        except Exception:
            seed_hash = None

    main_url = referer if referer.startswith("https://graph.baidu.com/s?") else build_graph_page(query_sign, query_image_url, pn=0, rn=60)
    context = await prepare_similar_context(session, main_url, debug=debug)
    if not context:
        if debug:
            print(f"[debug] failed to prepare similar context for {image_path.name}")
        print(f"{image_path.name} -> 0 images (saved to {slug})")
        return
    context.setdefault("sign", query_sign)

    ajax_params = context.get("ajax_params") if isinstance(context, dict) else None
    # ajax_params 目前仅用于确认接口存在；分页步长由远端控制
    dhash_min_distance = max(0, min(dhash_min_distance, dhash_threshold))

    seen_urls: set[str] = set()
    downloaded = 0
    tasks: list[tuple[str, str, asyncio.Task]] = []
    page_index = 0

    pages_without_progress = 0

    async def flush_pending_tasks() -> None:
        nonlocal downloaded, tasks
        if not tasks:
            return
        results = await asyncio.gather(*(t for _, _, t in tasks))
        for (url_cur, _, _), result in zip(tasks, results):
            got, normalized, _ = result
            if not got:
                continue
            if phash_filter and seed_hash is not None and normalized:
                try:
                    with Image.open(BytesIO(normalized)) as imd:
                        h = dhash(imd)
                        dist = hamming(seed_hash, h)
                        if dist <= dhash_min_distance:
                            sha1 = hashlib.sha1(normalized).hexdigest()
                            for p in [out_dir / f"{sha1}.jpg", out_dir / f"{sha1}.png", out_dir / f"{sha1}.json"]:
                                if p.exists():
                                    p.unlink(missing_ok=True)
                            if debug:
                                print(f"[debug] drop by dHash near-duplicate dist={dist} url={url_cur}")
                            continue
                        if dist > dhash_threshold:
                            sha1 = hashlib.sha1(normalized).hexdigest()
                            for p in [out_dir / f"{sha1}.jpg", out_dir / f"{sha1}.png", out_dir / f"{sha1}.json"]:
                                if p.exists():
                                    p.unlink(missing_ok=True)
                            if debug:
                                print(f"[debug] drop by dHash dist={dist} url={url_cur}")
                            continue
                except Exception:
                    pass
            downloaded += 1
            if downloaded >= limit:
                break
        tasks.clear()

    while downloaded < limit:
        start_downloaded = downloaded
        data = await fetch_similar_page(
            session,
            context,
            page_index,
            referer=referer,
            debug=debug,
        )
        page_index += 1
        entries = [e for e in data.get("data", []) if isinstance(e, dict) and e] if isinstance(data, dict) else []
        if not entries:
            if debug:
                print(f"[debug] no entries seed={image_path.name} page={page_index}")
            # 连续拿不到数据就收尾
            break

        batch_urls: list[tuple[str, str]] = []
        for entry in entries:
            url = entry.get("middleURL") or entry.get("objURL") or entry.get("hoverURL") or entry.get("thumbURL")
            if not isinstance(url, str):
                continue
            if not url.startswith(("http://", "https://")):
                continue
            url_key = GlobalDedupIndex.make_url_key(url)
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            claimed = True
            if dedup:
                claimed = await dedup.claim_url(url_key)
            if not claimed:
                continue
            batch_urls.append((url, url_key))

        if debug:
            print(f"[debug] seed={image_path.name} page={page_index} candidates={len(batch_urls)}")

        # 下载并（可选）做相似度过滤
        for url, url_key in batch_urls:
            task = asyncio.create_task(
                download_image(
                    session,
                    url,
                    out_dir,
                    dedup=dedup,
                    url_key=url_key,
                    referer=referer,
                    debug=debug,
                )
            )
            tasks.append((url, url_key, task))
            if len(tasks) >= batch_size:
                await flush_pending_tasks()
                if downloaded >= limit:
                    break

        if downloaded >= limit:
            break
        if tasks and downloaded < limit:
            await flush_pending_tasks()
            if downloaded >= limit:
                break

        if downloaded == start_downloaded:
            pages_without_progress += 1
        else:
            pages_without_progress = 0

        if pages_without_progress >= 20:
            if debug:
                print(f"[debug] stop seed={image_path.name} due to no progress after {pages_without_progress} pages")
            break

    rel_dir = _relative_to_root(out_dir, dest_root)
    print(f"{image_path.name} -> {downloaded} images (saved to {rel_dir})")


async def run_all(
    seeds: Iterable[pathlib.Path],
    *,
    dest_root: pathlib.Path,
    limit_per_seed: int,
    per_host: int,
    batch_size: int,
    max_concurrent: int,
    debug: bool,
    use_timestamp_subdir: bool,
    phash_filter: bool,
    dhash_threshold: int,
    dhash_min_distance: int,
    enable_dedup: bool,
) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    dedup = GlobalDedupIndex(dest_root) if enable_dedup else None

    connector = aiohttp.TCPConnector(limit_per_host=per_host, ssl=False)
    timeout = aiohttp.ClientTimeout(total=60, connect=15, sock_read=30)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def warm_up_root() -> None:
        for url in [
            "https://image.baidu.com/",
            "https://graph.baidu.com/pcpage/index?tpl_from=pc",
        ]:
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    await resp.read()
            except Exception:
                continue

    async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
        await warm_up_root()
        tasks = []

        async def worker(path: pathlib.Path) -> None:
            async with sem:
                await crawl_reference_image(
                    session,
                    path,
                    dest_root=dest_root,
                    limit=limit_per_seed,
                    batch_size=batch_size,
                    dedup=dedup,
                    use_timestamp_subdir=use_timestamp_subdir,
                    phash_filter=phash_filter,
                    dhash_threshold=dhash_threshold,
                    dhash_min_distance=dhash_min_distance,
                    debug=debug,
                )

        for seed in seeds:
            tasks.append(asyncio.create_task(worker(seed)))

        if tasks:
            await asyncio.gather(*tasks)


# ======================= CLI =======================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baidu 以图搜图抓取脚本（2025 版）")
    parser.add_argument("-i", "--input-dir", default="crawledimg/search_by_images/seed", help="种子图片目录")
    parser.add_argument("--glob", help="仅选择匹配该 glob 的文件（相对于 --input-dir）。")
    parser.add_argument("-o", "--output", default="crawledimg/search_by_images/baidu", help="下载结果根目录")
    parser.add_argument("-n", "--limit-per-seed", type=int, default=150, help="每张种子图片最多抓取的图片数量")
    parser.add_argument("--per-host", type=int, default=6, help="每主机并发上限")
    parser.add_argument("--batch-size", type=int, default=16, help="下载任务批量大小")
    parser.add_argument("--max-concurrent", type=int, default=3, help="同时处理的种子图片数量")
    parser.add_argument("--debug", action="store_true", help="输出调试信息并放宽图片尺寸过滤")
    parser.add_argument("-t", "--timestamp-subdir", action="store_true", help="为每个种子创建带时间戳的子目录保存结果")
    parser.add_argument("--no-phash", action="store_true", help="关闭 dHash 相似度筛选（默认开启）")
    parser.add_argument("--no-dedup", action="store_true", help="关闭跨运行去重（默认开启，会写入 _global_index.jsonl）")
    parser.add_argument("--dhash-threshold", type=int, default=DEFAULT_DHASH_THRESHOLD, help="dHash 筛选阈值（默认 18）")
    parser.add_argument("--dhash-min-distance", type=int, default=DEFAULT_DHASH_MIN_DISTANCE, help="dHash 最小距离，过小视为重复（默认 4）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_root = pathlib.Path(args.input_dir).resolve()
    dest_root = pathlib.Path(args.output).resolve()
    seeds = gather_seed_images(seed_root, args.glob)
    if not seeds:
        print(f"未找到任何种子图片：{seed_root}")
        return

    try:
        asyncio.run(
            run_all(
                seeds,
                dest_root=dest_root,
                limit_per_seed=args.limit_per_seed,
                per_host=args.per_host,
                batch_size=args.batch_size,
                max_concurrent=args.max_concurrent,
                debug=args.debug,
                use_timestamp_subdir=args.timestamp_subdir,
                phash_filter=not args.no_phash,
                dhash_threshold=args.dhash_threshold,
                dhash_min_distance=args.dhash_min_distance,
                enable_dedup=not args.no_dedup,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()
