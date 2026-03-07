#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PREFIXES = ("eidos_mcp_backup_",)
GLOBAL_TRIGGERS = {
    ".github",
    "lib",
}
GLOBAL_TRIGGER_FILES = {
    "pyproject.toml",
    "pytest.ini",
    "ruff.toml",
    "mypy.ini",
    "setup.py",
    "setup.cfg",
}


def _tests_dirs(root: Path) -> list[str]:
    out: list[str] = []
    for path in sorted(root.glob("*/tests")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        out.append(rel)
    return out


def _component_for(test_dir: str) -> str:
    return test_dir.split("/", 1)[0]


def _load_changed(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_matrix(changed_files: list[str], force_all: bool = False) -> dict[str, list[dict[str, str]]]:
    tests_dirs = _tests_dirs(REPO_ROOT)
    if force_all or not changed_files:
        return {"include": [{"component": _component_for(t), "test_path": t} for t in tests_dirs]}

    include_all = False
    touched_scopes: set[str] = set()
    for rel in changed_files:
        path = rel.strip()
        if path.startswith("./"):
            path = path[2:]
        if not path:
            continue
        first = path.split("/", 1)[0]
        if first in GLOBAL_TRIGGERS or path in GLOBAL_TRIGGER_FILES:
            include_all = True
            break
        if "/" not in path:
            # root-level changes outside docs tend to affect shared behavior
            if not path.endswith(".md"):
                include_all = True
                break
            continue
        touched_scopes.add(first)
        if first == "scripts" and path.startswith("scripts/tests/"):
            touched_scopes.add("scripts")
        if first == "benchmarks":
            touched_scopes.add("benchmarks")

    if include_all:
        return {"include": [{"component": _component_for(t), "test_path": t} for t in tests_dirs]}

    selected = []
    for test_dir in tests_dirs:
        component = _component_for(test_dir)
        if component in touched_scopes:
            selected.append({"component": component, "test_path": test_dir})

    if not selected:
        # Default to the most central regression surfaces if changes were documentation-only or ambiguous.
        fallback = [
            {"component": "agent_forge", "test_path": "agent_forge/tests"},
            {"component": "code_forge", "test_path": "code_forge/tests"},
            {"component": "knowledge_forge", "test_path": "knowledge_forge/tests"},
            {"component": "memory_forge", "test_path": "memory_forge/tests"},
            {"component": "eidos_mcp", "test_path": "eidos_mcp/tests"},
            {"component": "scripts", "test_path": "scripts/tests"},
            {"component": "web_interface_forge", "test_path": "web_interface_forge/tests"},
        ]
        return {"include": fallback}

    selected.sort(key=lambda item: item["component"])
    return {"include": selected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-files", default="", help="Path to newline-delimited changed-files list")
    parser.add_argument("--all", action="store_true", help="Force full matrix")
    args = parser.parse_args()

    changed = _load_changed(Path(args.changed_files)) if args.changed_files else []
    payload = build_matrix(changed, force_all=args.all)
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
