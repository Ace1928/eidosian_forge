#!/usr/bin/env python3
"""Quarantine sanitized Moltbook payloads with decision metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import sys
from pathlib import Path
from typing import Iterable

from moltbook_forge.moltbook_screen import main as screen_main


def _load_payload(path: str) -> dict:
    if path == "-":
        return json.loads(sys.stdin.read())
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_stem(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _decision_for(payload_path: Path, threshold: float) -> dict:
    tmp_output = payload_path.parent / ".decision.tmp.json"
    try:
        screen_main(
            [
                "--input",
                str(payload_path),
                "--output",
                str(tmp_output),
                "--threshold",
                str(threshold),
            ]
        )
        return json.loads(tmp_output.read_text(encoding="utf-8"))
    finally:
        if tmp_output.exists():
            tmp_output.unlink()


def _quarantine_dir(path: str) -> Path:
    base = Path(path).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quarantine Moltbook payloads based on screening decision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Sanitized JSON input or '-' for stdin")
    parser.add_argument("--quarantine-dir", default="moltbook_quarantine", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.4, help="Risk score threshold")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    payload = _load_payload(args.input)
    with tempfile.TemporaryDirectory(prefix="moltbook_tmp_") as tmp_name:
        tmp_dir = Path(tmp_name)
        tmp_payload = tmp_dir / "payload.json"
        _write_json(tmp_payload, payload)
        decision = _decision_for(tmp_payload, args.threshold)

    if decision.get("decision") != "quarantine":
        print(json.dumps({"decision": "allow", "reason": decision.get("reason", "")}))
        return 0

    text = payload.get("text", "")
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    stamp = int(time.time())
    stem = _safe_stem(f"quarantine_{stamp}_{digest}")
    filename = f"{stem}.json"
    out_dir = _quarantine_dir(args.quarantine_dir)
    out_path = out_dir / filename
    _write_json(out_path, {"payload": payload, "decision": decision})
    print(json.dumps({"decision": "quarantine", "path": str(out_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
