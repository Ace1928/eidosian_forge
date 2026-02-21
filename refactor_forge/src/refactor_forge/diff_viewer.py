"""Diff preview utilities for refactor dry-run workflows."""

from __future__ import annotations

import difflib
from pathlib import Path


def unified_diff_text(
    before: str,
    after: str,
    *,
    fromfile: str = "before.py",
    tofile: str = "after.py",
) -> str:
    """Build a unified diff string from two source texts."""
    diff_lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )
    return "\n".join(diff_lines)


def load_and_diff(source_path: Path, proposed_path: Path) -> str:
    """Load two files and return a unified diff."""
    source_text = source_path.read_text(encoding="utf-8")
    proposed_text = proposed_path.read_text(encoding="utf-8")
    return unified_diff_text(
        source_text,
        proposed_text,
        fromfile=str(source_path),
        tofile=str(proposed_path),
    )
