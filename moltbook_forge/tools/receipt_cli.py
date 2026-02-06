#!/usr/bin/env python3
"""CLI for proof-of-process receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from moltbook_forge.receipts import Artifact, ToolCall, build_receipt, load_receipt, save_receipt


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate proof-of-process receipts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    create = sub.add_parser("create", help="Create a new receipt")
    create.add_argument("--input", required=True, help="Input text file")
    create.add_argument("--output", required=True, help="Receipt output path")
    create.add_argument("--summary", default="", help="Optional summary")
    create.add_argument("--tools", default="", help="JSON file with tool calls")
    create.add_argument("--artifacts", default="", help="JSON file with artifacts")
    create.add_argument("--output-text", default="", help="Optional output text file")

    validate = sub.add_parser("validate", help="Validate a receipt file")
    validate.add_argument("--input", required=True, help="Receipt JSON file")
    return parser.parse_args(list(argv))


def _parse_tool_calls(path: Path | None) -> list[ToolCall]:
    if path is None:
        return []
    payload = _load_json(path)
    return [ToolCall.model_validate(item) for item in payload]


def _parse_artifacts(path: Path | None) -> list[Artifact]:
    if path is None:
        return []
    payload = _load_json(path)
    return [Artifact.model_validate(item) for item in payload]


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.cmd == "create":
        input_path = Path(args.input)
        output_path = Path(args.output)
        output_text = Path(args.output_text) if args.output_text else None
        tool_path = Path(args.tools) if args.tools else None
        artifact_path = Path(args.artifacts) if args.artifacts else None
        receipt = build_receipt(
            input_text=_read_text(input_path),
            input_path=str(input_path),
            summary=args.summary or None,
            tool_calls=_parse_tool_calls(tool_path),
            artifacts=_parse_artifacts(artifact_path),
            output_text=_read_text(output_text) if output_text else None,
        )
        save_receipt(str(output_path), receipt)
        print(f"Wrote receipt {receipt.receipt_id} to {output_path}")
        return 0
    if args.cmd == "validate":
        receipt = load_receipt(args.input)
        status = "VALID" if receipt.verify() else "INVALID"
        print(f"{status} receipt_id={receipt.receipt_id} content_hash={receipt.content_hash()}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
