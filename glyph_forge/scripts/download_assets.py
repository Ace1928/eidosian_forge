#!/usr/bin/env python3
"""Download asset library based on manifest.json."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "assets" / "library" / "manifest.json"
OUT_DIR = ROOT / "assets" / "library" / "downloads"
ATTRIBUTION = ROOT / "assets" / "library" / "ATTRIBUTION.md"
MAX_GB = float(os.environ.get("GLYPH_FORGE_ASSET_MAX_GB", "5"))


def _size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 ** 3)


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    if _size_gb(OUT_DIR) >= MAX_GB:
        return
    headers = {"User-Agent": "GlyphForgeAssetFetcher/1.0"}
    try:
        resp = requests.get(url, stream=True, timeout=60, headers=headers)
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    except requests.HTTPError:
        # Skip blocked assets and continue
        if out_path.exists():
            out_path.unlink()
        return


def _nasa_search(query: str, media_types: list[str], count: int) -> list[str]:
    urls: list[str] = []
    media_type = ",".join(media_types)
    base = "https://images-api.nasa.gov/search"
    page = 1
    while len(urls) < count:
        url = f"{base}?q={quote(query)}&media_type={quote(media_type)}&page={page}&page_size=100"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("collection", {}).get("items", [])
        if not items:
            break
        for item in items:
            data_items = item.get("data", [])
            nasa_id = data_items[0].get("nasa_id") if data_items else None
            if not nasa_id:
                continue
            asset_urls = _nasa_asset_urls(nasa_id)
            if not asset_urls:
                continue
            href = _pick_nasa_asset(asset_urls)
            if href and href not in urls:
                urls.append(href)
            if len(urls) >= count:
                break
        page += 1
    return urls


def _nasa_asset_urls(nasa_id: str) -> list[str]:
    asset_url = f"https://images-api.nasa.gov/asset/{nasa_id}"
    resp = requests.get(asset_url, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("collection", {}).get("items", [])
    urls = [item.get("href") for item in items if item.get("href")]
    # Prefer medium/large JPG or MP4
    preferred = []
    for u in urls:
        if u.endswith("~medium.jpg") or u.endswith("~large.jpg") or u.endswith(".jpg"):
            preferred.append(u)
    for u in urls:
        if u.endswith("~medium.mp4") or u.endswith("~large.mp4") or u.endswith(".mp4"):
            preferred.append(u)
    return preferred or urls


def _pick_nasa_asset(urls: list[str]) -> str | None:
    """Prefer highest-quality MP4 or JPG (orig > large > medium > small)."""
    if not urls:
        return None
    def _rank(u: str) -> tuple[int, int]:
        u_l = u.lower()
        # media priority: mp4 first, then jpg
        if u_l.endswith(".mp4"):
            media = 0
        elif u_l.endswith(".jpg"):
            media = 1
        else:
            media = 2
        # quality tokens
        if "orig" in u_l:
            qual = 0
        elif "~large" in u_l or "large" in u_l:
            qual = 1
        elif "~medium" in u_l or "medium" in u_l:
            qual = 2
        elif "~small" in u_l or "small" in u_l:
            qual = 3
        else:
            qual = 4
        return (media, qual)
    best = sorted(urls, key=_rank)[0]
    return best


def _confirm_action(message: str, assume_yes: bool | None = None, default: bool = False) -> bool:
    if assume_yes is True:
        return True
    if assume_yes is False:
        return False
    prompt = " [Y/n] " if default else " [y/N] "
    reply = input(message + prompt).strip().lower()
    if not reply:
        return default
    return reply.startswith("y")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts (assume yes)")
    parser.add_argument("--no", action="store_true", help="Skip confirmation prompts (assume no)")
    parser.add_argument("--max-gb", type=float, default=MAX_GB, help="Maximum download size in GB")
    args = parser.parse_args()
    global MAX_GB
    MAX_GB = args.max_gb

    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}")
        return 1

    manifest = json.loads(MANIFEST.read_text())
    sources = manifest.get("sources", [])
    if not _confirm_action(
        f"Download {len(sources)} source groups to {OUT_DIR}? Max size {args.max_gb} GB.",
        assume_yes=True if args.yes else False if args.no else None,
        default=False,
    ):
        print("Download canceled.")
        return 0
    attribution_lines = [
        "# Glyph Forge Asset Attribution",
        "",
        "This library contains public-domain and Creative Commons assets.",
        "See manifest.json for source details.",
        "",
    ]

    for entry in sources:
        if entry.get("type") == "nasa_search":
            query = entry.get("query", "")
            media_types = entry.get("media_types", ["image"])
            count = int(entry.get("count", 10))
            urls = _nasa_search(query, media_types, count)
            for idx, url in enumerate(urls, start=1):
                ext = url.split("?")[0].split(".")[-1]
                filename = f"nasa_{query.replace(' ', '_')}_{idx}.{ext}"
                _download(url, OUT_DIR / filename)
            attribution_lines.append(f"- NASA Image and Video Library search: {query}")
        elif entry.get("type") == "wikimedia_file":
            url = entry.get("download_url")
            if not url:
                continue
            filename = url.split("/")[-1]
            _download(url, OUT_DIR / filename)
            attribution_lines.append(f"- Wikimedia Commons: {entry.get('source_url')}")
        if _size_gb(OUT_DIR) >= args.max_gb:
            break

    ATTRIBUTION.write_text("\n".join(attribution_lines), encoding="utf-8")
    print(f"Assets downloaded to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
