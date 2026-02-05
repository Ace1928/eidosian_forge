#!/usr/bin/env python3
"""Normalize and flag untrusted Moltbook content.

Example:
  python moltbook_forge/moltbook_sanitize.py --input post.txt
  python moltbook_forge/moltbook_sanitize.py --input - --max-chars 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from typing import Iterable, List


ZERO_WIDTH = "".join(
    [
        "\u200b",
        "\u200c",
        "\u200d",
        "\u200e",
        "\u200f",
        "\u2060",
        "\ufeff",
    ]
)

SUSPICIOUS_PATTERNS = [
    r"ignore (all|any|previous) (instructions|rules)",
    r"system prompt",
    r"developer message",
    r"you are (chatgpt|an ai|a language model)",
    r"act as",
    r"jailbreak",
    r"do not (tell|reveal)",
    r"bypass",
    r"execute (code|commands?)",
    r"run (shell|bash|cmd|powershell)",
    r"curl ",
    r"wget ",
    r"pip install",
    r"rm -rf",
    r"sudo",
]


@dataclass
class NormalizedContent:
    raw_sha256: str
    normalized_sha256: str
    length_raw: int
    length_normalized: int
    truncated: bool
    flags: List[str]
    text: str


def _strip_zero_width(text: str) -> str:
    if not text:
        return text
    return text.translate({ord(ch): "" for ch in ZERO_WIDTH})


def _strip_control_chars(text: str) -> str:
    cleaned: List[str] = []
    for ch in text:
        if ch in ("\n", "\t"):
            cleaned.append(ch)
            continue
        if unicodedata.category(ch) == "Cc":
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_flags(text: str) -> List[str]:
    lowered = text.lower()
    flags = []
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, lowered):
            flags.append(pattern)
    return flags


def normalize_text(text: str, max_chars: int = 20000) -> NormalizedContent:
    raw = text or ""
    raw_hash = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()

    normalized = unicodedata.normalize("NFKC", raw)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _strip_zero_width(normalized)
    normalized = _strip_control_chars(normalized)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    normalized = _collapse_whitespace(normalized)

    truncated = False
    if len(normalized) > max_chars:
        normalized = normalized[:max_chars]
        truncated = True

    flags = _detect_flags(normalized)
    norm_hash = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
    return NormalizedContent(
        raw_sha256=raw_hash,
        normalized_sha256=norm_hash,
        length_raw=len(raw),
        length_normalized=len(normalized),
        truncated=truncated,
        flags=flags,
        text=normalized,
    )


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and flag untrusted Moltbook content",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input file path or '-' for stdin")
    parser.add_argument("--output", default="", help="Write JSON to file instead of stdout")
    parser.add_argument("--max-chars", type=int, default=20000, help="Maximum normalized length")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    content = _read_input(args.input)
    normalized = normalize_text(content, max_chars=args.max_chars)
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
