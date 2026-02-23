#!/usr/bin/env python3
"""Inventory open checklist items across TODO/PLAN/ROADMAP markdown documents."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

OPEN_PATTERN = re.compile(r"^\s*[-*]\s*\[ \]\s+(.*)\s*$")
DOC_NAME_PATTERN = re.compile(r"(todo|plan|roadmap)", re.IGNORECASE)

DEFAULT_EXCLUDE_SEGMENTS = {
    "archive_forge",
    "doc_forge/final_docs",
    "data/code_forge/roundtrip",
    "eidos_mcp_backup_20260218",
    ".gemini/tmp",
    "node_modules",
    ".git",
    "eidosian_venv",
}


@dataclass
class OpenItem:
    path: str
    line: int
    text: str


@dataclass
class FileSummary:
    path: str
    open_count: int


def iter_plan_docs(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.md"):
        if DOC_NAME_PATTERN.search(path.name):
            yield path


def is_excluded(path: Path, root: Path, extra_exclude: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    for seg in DEFAULT_EXCLUDE_SEGMENTS:
        if rel == seg or rel.startswith(seg + "/"):
            return True
    for seg in extra_exclude:
        seg = seg.strip().rstrip("/")
        if not seg:
            continue
        if rel == seg or rel.startswith(seg + "/"):
            return True
    return False


def collect(root: Path, extra_exclude: list[str]) -> tuple[list[FileSummary], list[OpenItem]]:
    summaries: list[FileSummary] = []
    items: list[OpenItem] = []

    for path in sorted(iter_plan_docs(root)):
        if is_excluded(path, root, extra_exclude):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        open_count = 0
        for idx, line in enumerate(lines, start=1):
            m = OPEN_PATTERN.match(line)
            if not m:
                continue
            open_count += 1
            items.append(OpenItem(path=rel, line=idx, text=m.group(1).strip()))
        summaries.append(FileSummary(path=rel, open_count=open_count))

    summaries = [s for s in summaries if s.open_count > 0]
    summaries.sort(key=lambda s: (-s.open_count, s.path))
    return summaries, items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root path")
    parser.add_argument("--top", type=int, default=50, help="Show top N files by open items")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional path prefix to exclude (repeatable)",
    )
    parser.add_argument("--json-out", default=None, help="Optional JSON report output path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summaries, items = collect(root, args.exclude)

    total_open = sum(s.open_count for s in summaries)
    print(f"open_files={len(summaries)} open_items={total_open}")
    for summary in summaries[: max(1, int(args.top))]:
        print(f"{summary.open_count:4d} {summary.path}")

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(root),
            "open_files": len(summaries),
            "open_items": total_open,
            "files": [asdict(s) for s in summaries],
            "items": [asdict(i) for i in items],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json_report={out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
