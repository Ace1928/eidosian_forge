#!/usr/bin/env python3
"""Fetch or read Moltbook content and emit normalized JSON."""

from __future__ import annotations

import os

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict
from typing import Iterable

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from moltbook_forge.moltbook_sanitize import normalize_text


ALLOWED_HOSTS = {"moltbook.com", "www.moltbook.com"}


def _fetch_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")
    host = parsed.netloc.split(":")[0].lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"Host '{host}' is not allowlisted")
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _read_file(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Moltbook content with strict normalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="", help="Fetch content from URL (requires --allow-network)")
    parser.add_argument("--file", default="", help="Read content from file path or '-' for stdin")
    parser.add_argument("--allow-network", action="store_true", help="Allow network fetch")
    parser.add_argument("--max-chars", type=int, default=20000, help="Maximum normalized length")
    parser.add_argument("--output", default="", help="Write JSON to file instead of stdout")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.url and not args.file:
        raise SystemExit("ERROR provide --url or --file")
    if args.url and not args.allow_network:
        raise SystemExit("ERROR --allow-network is required for URL ingestion")

    if args.url:
        raw = _fetch_url(args.url)
    else:
        raw = _read_file(args.file)

    normalized = normalize_text(raw, max_chars=args.max_chars)
    payload = json.dumps(asdict(normalized), indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
