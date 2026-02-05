#!/usr/bin/env python3
"""Validate Moltbook sanitized payload JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from moltbook_forge.moltbook_schema import validate_normalized_payload


def _load_payload(path: str) -> dict:
    if path == "-":
        return json.loads(sys.stdin.read())
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate sanitized Moltbook JSON payload",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Sanitized JSON input or '-' for stdin")
    parser.add_argument("--output", default="", help="Write validation JSON to file")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = _load_payload(args.input)
    result = validate_normalized_payload(payload)
    output = json.dumps({"ok": result.ok, "errors": result.errors}, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
            handle.write("\n")
    else:
        print(output)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
