#!/usr/bin/env python3
"""Review Moltbook skill content through the safety pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict
from typing import Iterable, List

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from moltbook_forge.moltbook_sanitize import normalize_text
from moltbook_forge.moltbook_schema import validate_normalized_payload
from moltbook_forge.moltbook_screen import screen_payload


ALLOWED_HOSTS = {"moltbook.com", "www.moltbook.com"}

COMMAND_PATTERNS = [
    r"^\s*\$\s+.+",
    r"^\s*(?:curl|wget|pip|python|python3|bash|sh|git|npm|node|cargo|make)\b.+",
]

SECRET_PATTERNS = [
    (re.compile(r'("api_key"\s*:\s*")[^"]+(")', re.IGNORECASE), r'\1***\2'),
    (re.compile(r'("moltbook_api_key"\s*:\s*")[^"]+(")', re.IGNORECASE), r'\1***\2'),
    (re.compile(r'(\bapi_key\b\s*[:=]\s*)\S+', re.IGNORECASE), r'\1***'),
    (re.compile(r'(\bmoltbook_api_key\b\s*[:=]\s*)\S+', re.IGNORECASE), r'\1***'),
    (re.compile(r'(\bauth(?:entication)?_?token\b\s*[:=]\s*)\S+', re.IGNORECASE), r'\1***'),
    (re.compile(r'(\bBearer\s+)[A-Za-z0-9._-]+', re.IGNORECASE), r'\1***'),
    (re.compile(r'\bclh_[A-Za-z0-9]+'), 'clh_***'),
    (re.compile(r'\bmoltbook_sk_[A-Za-z0-9]+'), 'moltbook_sk_***'),
    (re.compile(r'\bmoltdev_[A-Za-z0-9]+'), 'moltdev_***'),
]

def _fetch_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")
    host = parsed.netloc.split(":")[0].lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"Host '{host}' is not allowlisted")
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _extract_urls(text: str) -> List[str]:
    return sorted(set(re.findall(r"https?://\S+", text)))


def _extract_env(text: str) -> List[str]:
    matches = re.findall(r"\b[A-Z][A-Z0-9_]{2,}=[^\s]+", text)
    return sorted(set(matches))


def _extract_commands(text: str) -> List[str]:
    lines = text.splitlines()
    commands: List[str] = []
    for line in lines:
        for pattern in COMMAND_PATTERNS:
            if re.search(pattern, line):
                commands.append(line.strip())
                break
    return sorted(set(commands))


def _redact_secrets(text: str) -> str:
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review Moltbook skill content safely",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="", help="Input file path or '-' for stdin")
    parser.add_argument("--url", default="", help="Fetch content from URL (requires --allow-network)")
    parser.add_argument("--allow-network", action="store_true", help="Allow network fetch")
    parser.add_argument("--max-chars", type=int, default=20000, help="Maximum normalized length")
    parser.add_argument("--threshold", type=float, default=0.2, help="Risk score threshold")
    parser.add_argument("--include-payload", action="store_true", help="Include sanitized payload in report")
    parser.add_argument("--redact-secrets", action="store_true", default=True, help="Redact secrets in output")
    parser.add_argument("--no-redact-secrets", dest="redact_secrets", action="store_false", help="Disable redaction")
    parser.add_argument("--output", default="", help="Write report JSON to file instead of stdout")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.input and not args.url:
        raise SystemExit("ERROR provide --input or --url")
    if args.url and not args.allow_network:
        raise SystemExit("ERROR --allow-network is required for URL ingestion")

    if args.url:
        raw = _fetch_url(args.url)
    else:
        raw = _read_input(args.input)

    normalized = normalize_text(raw, max_chars=args.max_chars)
    redacted_text = _redact_secrets(normalized.text) if args.redact_secrets else normalized.text
    payload = asdict(normalized)
    validation = validate_normalized_payload(payload)
    decision = screen_payload(payload, args.threshold)

    report = {
        "validation_ok": validation.ok,
        "validation_errors": validation.errors,
        "decision": decision.decision,
        "risk_score": decision.risk_score,
        "flags": decision.flags,
        "reason": decision.reason,
        "extracted": {
            "commands": _extract_commands(normalized.text),
            "urls": _extract_urls(normalized.text),
            "env": _extract_env(normalized.text),
        },
        "redacted": args.redact_secrets,
        "redacted_text": redacted_text,
    }
    if args.include_payload:
        payload["text"] = redacted_text
        report["payload"] = payload

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
            handle.write("\n")
    else:
        print(output)

    if not validation.ok:
        return 1
    if decision.decision == "quarantine":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
