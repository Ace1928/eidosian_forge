#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

README_CANDIDATES = ("README.md", "README.rst", "readme.md", "Readme.md")
KEEP_HIDDEN_TOP = {".github", ".vscode", ".bin"}


def _run_git_ls_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=False,
        check=False,
    )
    if proc.returncode != 0:
        return []
    raw = proc.stdout.decode("utf-8", errors="replace")
    return [p for p in raw.split("\x00") if p]


def _iter_parent_dirs(path: str) -> Iterable[str]:
    parent = Path(path).parent
    while str(parent) not in {"", "."}:
        yield parent.as_posix()
        parent = parent.parent


def _tracked_dirs_from_files(tracked_files: list[str]) -> list[str]:
    dirs: set[str] = set()
    for file_path in tracked_files:
        for parent in _iter_parent_dirs(file_path):
            dirs.add(parent)
    return sorted(dirs, key=lambda p: (p.split("/", 1)[0], p.count("/"), p.lower()))


def _collect_filesystem_dirs(root: Path, *, include_hidden_top: bool) -> list[str]:
    dirs: set[str] = set()
    for current_root, subdirs, _files in os.walk(root):
        current_rel = Path(current_root).resolve().relative_to(root).as_posix()
        if current_rel == ".":
            current_rel = ""

        filtered: list[str] = []
        for name in subdirs:
            if name == ".git":
                continue
            if current_rel == "" and name.startswith("."):
                if not include_hidden_top and name not in KEEP_HIDDEN_TOP:
                    continue
            filtered.append(name)

        subdirs[:] = filtered
        for name in filtered:
            rel = f"{current_rel}/{name}" if current_rel else name
            dirs.add(rel)

    return sorted(dirs, key=lambda p: (p.split("/", 1)[0], p.count("/"), p.lower()))


def _runtime_top_level_dirs(root: Path, *, include_hidden_top: bool) -> list[str]:
    out: list[str] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if child.name == ".git":
            continue
        if child.name.startswith(".") and not include_hidden_top and child.name not in KEEP_HIDDEN_TOP:
            continue
        out.append(child.name)
    return out


def _readme_for_dir(root: Path, rel_dir: str, tracked_files: set[str] | None = None) -> str:
    base = root / rel_dir
    for name in README_CANDIDATES:
        candidate = base / name
        rel_path = f"{rel_dir}/{name}" if rel_dir else name
        if tracked_files is not None and rel_path not in tracked_files:
            continue
        if candidate.exists() and candidate.is_file():
            return rel_path
    return ""


def _category(rel_dir: str) -> str:
    if rel_dir.startswith(".github"):
        return "CI/CD automation"
    if rel_dir.startswith("docs"):
        return "Documentation"
    if rel_dir.startswith("scripts"):
        return "Automation scripts"
    if rel_dir.startswith("reports"):
        return "Benchmark/audit reports"
    if rel_dir.startswith("state"):
        return "Runtime state"
    if rel_dir.startswith("data"):
        return "Data and memory artifacts"
    if rel_dir.startswith("requirements"):
        return "Dependency manifests"
    if rel_dir.startswith("lib"):
        return "Shared libraries"
    if rel_dir.startswith("bin"):
        return "Top-level CLI entrypoints"
    if rel_dir.startswith("eidosian_venv"):
        return "Local virtual environment"
    if rel_dir.startswith("Backups"):
        return "Local backups"
    if rel_dir.startswith("logs"):
        return "Runtime logs"

    top = rel_dir.split("/", 1)[0]
    parts = rel_dir.split("/")
    base = parts[-1]

    if top.endswith("_forge") and len(parts) == 1:
        return "Forge module root"
    if base == "src":
        return "Source code"
    if base == "tests":
        return "Test suite"
    if base == "docs":
        return "Module docs"
    if base == "scripts":
        return "Module scripts"
    if base == "bin":
        return "Executable scripts"
    if base == "completions":
        return "Shell completions"
    if base == "examples":
        return "Examples"
    if base == "benchmarks":
        return "Benchmarks"
    if base == "templates":
        return "Templates"
    if base == "tools":
        return "Developer tools"
    if base == "config":
        return "Configuration"
    if base == "ci":
        return "CI helpers"
    if base == "shared":
        return "Shared assets"

    return "Project directory"


def _md_link(rel_from_docs: str, label: str | None = None) -> str:
    target = f"../{rel_from_docs}"
    text = label or rel_from_docs
    return f"[`{text}`]({target})"


def _resolve_generated_at(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.lower() == "now":
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return candidate


def build_atlas(
    repo_root: Path,
    all_dirs: list[str],
    tracked_files: list[str],
    *,
    max_depth: int,
    scope: str,
    generated_at: str | None,
) -> str:
    visible_dirs = [d for d in all_dirs if (d.count("/") + 1) <= max_depth]
    tracked_file_set = set(tracked_files) if scope == "tracked" else None

    tracked_file_count: dict[str, int] = defaultdict(int)
    for file_path in tracked_files:
        for parent in _iter_parent_dirs(file_path):
            tracked_file_count[parent] += 1

    direct_subdirs: dict[str, set[str]] = defaultdict(set)
    for d in all_dirs:
        parent = str(Path(d).parent)
        if parent in {"", "."}:
            continue
        direct_subdirs[parent].add(d)

    top_levels = sorted({d.split("/", 1)[0] for d in all_dirs})
    lines: list[str] = []
    lines.append("# Eidosian Forge Directory Atlas")
    lines.append("")
    lines.append("> User-focused inventory with linked paths and purpose labels.")
    lines.append("")
    if generated_at:
        lines.append(f"- Generated: `{generated_at}`")
    else:
        lines.append("- Generated: `deterministic (timestamp omitted)`")
    lines.append(f"- Scope: `{scope}`")
    lines.append(f"- Tracked files: `{len(tracked_files)}`")
    lines.append(f"- Full directory count (scope): `{len(all_dirs)}`")
    lines.append(f"- Atlas directory count (depth <= {max_depth}): `{len(visible_dirs)}`")
    lines.append("- Full recursive index: [`docs/DIRECTORY_INDEX_FULL.txt`](./DIRECTORY_INDEX_FULL.txt)")
    lines.append("")

    lines.append("## Jump Table")
    lines.append("")
    for top in top_levels:
        anchor = top.lower().replace("_", "-").replace(".", "")
        lines.append(f"- [`{top}`](#{anchor})")
    lines.append("")

    lines.append("## Top-Level Overview")
    lines.append("")
    lines.append("| Directory | Category | Tracked files | Direct subdirs | README |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for top in top_levels:
        readme = _readme_for_dir(repo_root, top, tracked_file_set)
        readme_cell = _md_link(readme, readme) if readme else "-"
        lines.append(
            "| "
            + _md_link(top, top)
            + f" | {_category(top)} | {tracked_file_count.get(top, 0)} | {len(direct_subdirs.get(top, set()))} | {readme_cell} |"
        )
    lines.append("")

    lines.append("## Directory Inventory (Depth-Limited)")
    lines.append("")
    for top in top_levels:
        group = [d for d in visible_dirs if d == top or d.startswith(f"{top}/")]
        if not group:
            continue
        lines.append(f"### `{top}`")
        lines.append("")
        lines.append("| Path | Category | Tracked files | Direct subdirs | README |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for d in group:
            readme = _readme_for_dir(repo_root, d, tracked_file_set)
            readme_cell = _md_link(readme, readme) if readme else "-"
            lines.append(
                "| "
                + _md_link(d, d)
                + f" | {_category(d)} | {tracked_file_count.get(d, 0)} | {len(direct_subdirs.get(d, set()))} | {readme_cell} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_full_index(
    full_index_path: Path,
    all_dirs: list[str],
    *,
    scope: str,
    generated_at: str | None,
) -> None:
    lines = ["# Full Recursive Directory Index", ""]
    if generated_at:
        lines.append(f"Generated: {generated_at}")
    else:
        lines.append("Generated: deterministic (timestamp omitted)")
    lines.append(f"Scope: {scope}")
    lines.append(f"Directory count: {len(all_dirs)}")
    lines.append("")
    for d in all_dirs:
        lines.append(d)
    full_index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate repository directory atlas markdown")
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--atlas-output", default="docs/DIRECTORY_ATLAS.md", help="Atlas markdown path")
    parser.add_argument(
        "--full-output", default="docs/DIRECTORY_INDEX_FULL.txt", help="Full recursive directory index path"
    )
    parser.add_argument("--max-depth", type=int, default=2, help="Max directory depth for atlas inventory")
    parser.add_argument(
        "--scope",
        choices=("tracked", "filesystem"),
        default="tracked",
        help="Directory scope: tracked (deterministic repo scope) or filesystem (local runtime scope)",
    )
    parser.add_argument(
        "--include-runtime-top-level",
        action="store_true",
        help="When scope=tracked, include detected top-level filesystem directories",
    )
    parser.add_argument(
        "--include-hidden-top-level",
        action="store_true",
        help="Include hidden top-level directories (except .git, which is always excluded)",
    )
    parser.add_argument(
        "--generated-at",
        default="",
        help="Optional generation timestamp text. Use 'now' for current UTC. Omit for deterministic output.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    atlas_out = (root / args.atlas_output).resolve()
    full_out = (root / args.full_output).resolve()
    atlas_out.parent.mkdir(parents=True, exist_ok=True)
    full_out.parent.mkdir(parents=True, exist_ok=True)
    generated_at = _resolve_generated_at(str(args.generated_at))

    tracked_files = _run_git_ls_files(root)
    tracked_dirs = _tracked_dirs_from_files(tracked_files)

    if args.scope == "filesystem":
        all_dirs = _collect_filesystem_dirs(
            root,
            include_hidden_top=bool(args.include_hidden_top_level),
        )
    else:
        dir_set = set(tracked_dirs)
        if args.include_runtime_top_level:
            for top in _runtime_top_level_dirs(
                root,
                include_hidden_top=bool(args.include_hidden_top_level),
            ):
                dir_set.add(top)
        all_dirs = sorted(dir_set, key=lambda p: (p.split("/", 1)[0], p.count("/"), p.lower()))

    atlas_md = build_atlas(
        root,
        all_dirs,
        tracked_files,
        max_depth=max(1, int(args.max_depth)),
        scope=str(args.scope),
        generated_at=generated_at,
    )
    atlas_out.write_text(atlas_md, encoding="utf-8")
    write_full_index(
        full_out,
        all_dirs,
        scope=str(args.scope),
        generated_at=generated_at,
    )

    print(f"Directory atlas written: {atlas_out}")
    print(f"Full recursive index written: {full_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
