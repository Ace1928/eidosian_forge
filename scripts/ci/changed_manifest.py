#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PYTHON_COMPONENTS = [
    {"component": "agent_forge", "root": "agent_forge", "test_path": "agent_forge/tests"},
    {"component": "benchmarks", "root": "benchmarks", "test_path": "benchmarks/tests"},
    {"component": "code_forge", "root": "code_forge", "test_path": "code_forge/tests"},
    {"component": "crawl_forge", "root": "crawl_forge", "test_path": "crawl_forge/tests"},
    {"component": "doc_forge", "root": "doc_forge", "test_path": "doc_forge/src/doc_forge/scribe/tests"},
    {"component": "eidos_mcp", "root": "eidos_mcp", "test_path": "eidos_mcp/tests"},
    {"component": "knowledge_forge", "root": "knowledge_forge", "test_path": "knowledge_forge/tests"},
    {"component": "lib", "root": "lib", "test_path": "lib/tests"},
    {"component": "llm_forge", "root": "llm_forge", "test_path": "llm_forge/tests"},
    {"component": "memory_forge", "root": "memory_forge", "test_path": "memory_forge/tests"},
    {"component": "scripts", "root": "scripts", "test_path": "scripts/tests"},
    {"component": "tests", "root": "tests", "test_path": "tests"},
    {"component": "web_interface_forge", "root": "web_interface_forge", "test_path": "web_interface_forge/tests"},
    {"component": "word_forge", "root": "word_forge", "test_path": "word_forge/tests"},
]

PYTHON_GLOBAL_PREFIXES = {
    "lib",
}
PYTHON_TRIGGER_FILES = {
    "mypy.ini",
    "pyproject.toml",
    "pytest.ini",
    "ruff.toml",
    "setup.cfg",
    "setup.py",
}
PRETTIER_EXTENSIONS = {".js", ".jsx", ".json", ".ts", ".tsx", ".yaml", ".yml"}
PYTHON_EXTENSIONS = {".py"}
IGNORED_PREFIXES = (
    ".git/",
    "data/bench_workspaces/",
    "data/runtime/",
    "docs/external_references/",
    "eidosian_venv/",
    "node_modules/",
    "reports/",
)
AUTSEED_ROOT = "game_forge/src/autoseed"


def _normalize(path: str) -> str:
    cleaned = path.strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _is_ignored(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in IGNORED_PREFIXES)


def _tracked_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [_normalize(line) for line in proc.stdout.splitlines() if _normalize(line)]


def _load_changed(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [_normalize(line) for line in path.read_text(encoding="utf-8").splitlines() if _normalize(line)]


def _filter_files(paths: list[str], extensions: set[str]) -> list[str]:
    out: list[str] = []
    for rel in paths:
        if _is_ignored(rel):
            continue
        if Path(rel).suffix not in extensions:
            continue
        if rel not in out:
            out.append(rel)
    return out


def _python_component_payload(force_all: bool, changed_files: list[str]) -> dict[str, list[dict[str, str]]]:
    if force_all or not changed_files:
        return {"include": PYTHON_COMPONENTS}

    known_components = {entry["component"] for entry in PYTHON_COMPONENTS}
    touched: set[str] = set()
    include_all = False

    for rel in changed_files:
        if _is_ignored(rel):
            continue
        first = rel.split("/", 1)[0]
        if first in PYTHON_GLOBAL_PREFIXES or rel in PYTHON_TRIGGER_FILES:
            include_all = True
            break
        if "/" not in rel:
            if not rel.endswith(".md"):
                include_all = True
                break
            continue
        if first in known_components:
            touched.add(first)
        elif rel.startswith("doc_forge/src/doc_forge/scribe/"):
            touched.add("doc_forge")
        elif rel.startswith("game_forge/src/autoseed/"):
            # Keep TS-only project out of Python matrix.
            continue

    if include_all:
        return {"include": PYTHON_COMPONENTS}

    selected = [entry for entry in PYTHON_COMPONENTS if entry["component"] in touched]
    return {"include": selected}


def _typescript_project_payload(force_all: bool, changed_files: list[str]) -> dict[str, list[dict[str, str]]]:
    if force_all:
        return {"include": [{"project": "autoseed", "root": AUTSEED_ROOT}]}

    include = any(rel.startswith(f"{AUTSEED_ROOT}/") for rel in changed_files if not _is_ignored(rel))
    if include:
        return {"include": [{"project": "autoseed", "root": AUTSEED_ROOT}]}
    return {"include": []}


def build_manifest(changed_files: list[str], force_all: bool = False) -> dict[str, object]:
    source_files = _tracked_files() if force_all else changed_files
    python_files = _filter_files(source_files, PYTHON_EXTENSIONS)
    prettier_files = _filter_files(source_files, PRETTIER_EXTENSIONS)
    python_components = _python_component_payload(force_all, changed_files)
    typescript_projects = _typescript_project_payload(force_all, changed_files)
    return {
        "python_files": python_files,
        "prettier_files": prettier_files,
        "python_components": python_components,
        "typescript_projects": typescript_projects,
        "has_python_files": bool(python_files),
        "has_prettier_files": bool(prettier_files),
        "has_typescript_projects": bool(typescript_projects["include"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-files", default="", help="Path to newline-delimited changed-files list")
    parser.add_argument("--all", action="store_true", help="Force a full manifest from tracked files")
    args = parser.parse_args()

    changed = _load_changed(Path(args.changed_files)) if args.changed_files else []
    print(json.dumps(build_manifest(changed, force_all=args.all), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
