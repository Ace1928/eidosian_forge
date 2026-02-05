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
    r"download .* and (run|execute)",
    r"fetch (?:.* )?instructions",
    r"remote instructions",
    r"heartbeat",
    r"check in every",
    r"every \\d+ hours",
    r"\|\s*(bash|sh)",
    r"curl ",
    r"wget ",
    r"npx ",
    r"pip install",
    r"rm -rf",
    r"sudo",
    r"api key",
    r"auth token",
]


@dataclass
class NormalizedContent:
    raw_sha256: str
    normalized_sha256: str
    length_raw: int
    length_normalized: int
    line_count: int
    word_count: int
    non_ascii_ratio: float
    truncated: bool
    flags: List[str]
    risk_score: float
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


def _detect_base64(text: str) -> bool:
    return re.search(r"(?:[A-Za-z0-9+/]{120,}={0,2})", text) is not None


def _detect_urls(text: str) -> bool:
    return re.search(r"https?://\\S+", text) is not None


def _detect_repeated_lines(text: str) -> bool:
    counts = {}
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 12:
            continue
        counts[line] = counts.get(line, 0) + 1
        if counts[line] >= 3:
            return True
    return False


def _detect_flags(text: str) -> List[str]:
    lowered = text.lower()
    flags = []
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, lowered):
            flags.append(pattern)
    if "```" in text:
        flags.append("code_fence")
    if _detect_base64(text):
        flags.append("base64_blob")
    if _detect_urls(text):
        flags.append("contains_url")
    if _detect_repeated_lines(text):
        flags.append("repeated_lines")
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
    length_norm = len(normalized)
    line_count = len(normalized.splitlines()) if normalized else 0
    word_count = len(re.findall(r"\b\w+\b", normalized))
    non_ascii = sum(1 for ch in normalized if ord(ch) > 127)
    non_ascii_ratio = non_ascii / max(1, length_norm)
    if non_ascii_ratio > 0.2:
        flags.append("high_non_ascii_ratio")

    critical_patterns = {
        r"ignore (all|any|previous) (instructions|rules)",
        r"system prompt",
        r"developer message",
        r"jailbreak",
        r"execute (code|commands?)",
        r"run (shell|bash|cmd|powershell)",
        r"download .* and (run|execute)",
        r"fetch (?:.* )?instructions",
        r"remote instructions",
        r"\|\s*(bash|sh)",
    }
    score = 0.0
    for flag in flags:
        score += 0.3 if flag in critical_patterns else 0.1
    if "base64_blob" in flags:
        score += 0.2
    if length_norm > 15000:
        score += 0.1
    score = min(1.0, score)

    norm_hash = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
    return NormalizedContent(
        raw_sha256=raw_hash,
        normalized_sha256=norm_hash,
        length_raw=len(raw),
        length_normalized=length_norm,
        line_count=line_count,
        word_count=word_count,
        non_ascii_ratio=round(non_ascii_ratio, 6),
        truncated=truncated,
        flags=flags,
        risk_score=round(score, 3),
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
