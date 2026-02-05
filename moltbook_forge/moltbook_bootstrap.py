#!/usr/bin/env python3
"""Bootstrap Moltbook content through the safety pipeline.

Example:
  python moltbook_forge/moltbook_bootstrap.py --input skill.md --output-dir moltbook_forge/skill_sources
  python moltbook_forge/moltbook_bootstrap.py --url https://www.moltbook.com/skill.md --allow-network
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from moltbook_forge.moltbook_schema import validate_normalized_payload
from moltbook_forge.moltbook_screen import screen_payload
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


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap Moltbook content through the safety pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="", help="Input file path or '-' for stdin")
    parser.add_argument("--url", default="", help="Fetch content from URL (requires --allow-network)")
    parser.add_argument("--allow-network", action="store_true", help="Allow network fetch")
    parser.add_argument("--max-chars", type=int, default=20000, help="Maximum normalized length")
    parser.add_argument("--threshold", type=float, default=0.4, help="Risk score threshold")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "skill_sources"),
        help="Directory for output artifacts",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.input and not args.url:
        raise SystemExit("ERROR provide --input or --url")
    if args.url and not args.allow_network:
        raise SystemExit("ERROR --allow-network is required for URL ingestion")

    if args.url:
        raw = _fetch_url(args.url)
        source = {"url": args.url}
    else:
        raw = _read_input(args.input)
        source = {"file": args.input}

    normalized = normalize_text(raw, max_chars=args.max_chars)
    payload = asdict(normalized)
    validation = validate_normalized_payload(payload)
    decision = screen_payload(payload, args.threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report = {
        "timestamp": timestamp,
        "source": source,
        "validation_ok": validation.ok,
        "validation_errors": validation.errors,
        "decision": decision.decision,
        "risk_score": decision.risk_score,
        "flags": decision.flags,
        "reason": decision.reason,
        "normalized_sha256": payload.get("normalized_sha256"),
        "raw_sha256": payload.get("raw_sha256"),
    }

    _write_json(output_dir / "sanitized.json", payload)
    _write_json(output_dir / "validation.json", {"ok": validation.ok, "errors": validation.errors})
    _write_json(
        output_dir / "decision.json",
        {
            "decision": decision.decision,
            "risk_score": decision.risk_score,
            "flags": decision.flags,
            "reason": decision.reason,
        },
    )
    _write_json(output_dir / "report.json", report)

    if decision.decision == "quarantine":
        _write_json(
            output_dir / "quarantine.json",
            {"payload": payload, "decision": decision.__dict__},
        )
        return 2
    if not validation.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
