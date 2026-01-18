#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_EXTS = [
    ".md",
    ".rst",
    ".txt",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
]
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
    ".var",
    "node_modules",
    "dist",
    "build",
    "site-packages",
    "target",
}
DEFAULT_EXCLUDE_GLOBS = [
    "*/.git/*",
    "*/.venv/*",
    "*/node_modules/*",
    "*/dist/*",
    "*/build/*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally index local text files with GraphRAG.",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="GraphRAG project root (contains settings.yaml, input/, output/).",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        type=Path,
        required=True,
        help="Root path to scan for text files. Repeatable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of files per incremental batch.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit number of batches processed (0 = no limit).",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[],
        help="File extension to include (repeatable). Defaults to common text/code.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=200_000,
        help="Skip files larger than this size in bytes (default: 200k).",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude (repeatable).",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Glob pattern to exclude (repeatable).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into input instead of symlinking.",
    )
    parser.add_argument(
        "--graphrag-cmd",
        default="graphrag",
        help="Command to invoke GraphRAG (default: graphrag).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose GraphRAG logging.",
    )
    parser.add_argument(
        "--method",
        choices=["standard", "fast"],
        default="fast",
        help="Indexing method to use (default: fast).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Path to state file (default: <root>/.local_index_state.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report work without running GraphRAG or writing input.",
    )
    return parser.parse_args()


def normalize_exts(exts: list[str]) -> list[str]:
    if not exts:
        return DEFAULT_EXTS
    normalized = []
    for ext in exts:
        ext = ext.strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext.lower())
    return normalized


def slugify_path(path: Path) -> str:
    slug = str(path.resolve())
    slug = slug.replace(os.sep, "_").replace(":", "")
    return slug.strip("_")


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"processed": {}, "indexed": False}


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def should_skip_dir(dirname: str, exclude_dirs: set[str]) -> bool:
    return dirname in exclude_dirs


def is_excluded(path: Path, exclude_globs: list[str]) -> bool:
    as_posix = path.as_posix()
    return any(fnmatch.fnmatch(as_posix, pattern) for pattern in exclude_globs)


def iter_files(
    scan_root: Path,
    exts: list[str],
    exclude_dirs: set[str],
    exclude_globs: list[str],
    max_bytes: int | None,
) -> list[Path]:
    collected = []
    for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d, exclude_dirs)]
        current_dir = Path(dirpath)
        if is_excluded(current_dir, exclude_globs):
            dirnames[:] = []
            continue
        for name in filenames:
            path = current_dir / name
            if is_excluded(path, exclude_globs):
                continue
            if path.suffix.lower() in exts:
                if max_bytes is not None:
                    try:
                        if path.stat().st_size > max_bytes:
                            continue
                    except (FileNotFoundError, PermissionError):
                        continue
                collected.append(path)
    return collected


def has_existing_output(root: Path) -> bool:
    output_dir = root / "output"
    if not output_dir.exists():
        return False
    required = output_dir / "entities.parquet"
    return required.exists()


def clear_input_dir(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for child in input_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def stage_files(
    input_dir: Path,
    scan_root: Path,
    files: list[Path],
    use_copy: bool,
) -> None:
    root_slug = slugify_path(scan_root)
    for src in files:
        rel = src.relative_to(scan_root)
        dest = input_dir / root_slug / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if use_copy:
            try:
                shutil.copy2(src, dest)
            except (FileNotFoundError, PermissionError):
                continue
        else:
            try:
                os.symlink(src, dest)
            except OSError:
                try:
                    shutil.copy2(src, dest)
                except (FileNotFoundError, PermissionError):
                    continue


def needs_index(path: Path, state: dict) -> bool:
    entry = state["processed"].get(str(path))
    try:
        stat = path.stat()
    except (FileNotFoundError, PermissionError):
        return False
    return (
        entry is None
        or entry.get("mtime") != stat.st_mtime
        or entry.get("size") != stat.st_size
    )


def run_graphrag(cmd: str, args: list[str]) -> None:
    cmd_parts = shlex.split(cmd) + args
    subprocess.run(cmd_parts, check=True)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 2
    input_dir = root / "input"

    state_path = args.state_file or (root / ".local_index_state.json")
    state = load_state(state_path)
    exts = normalize_exts(args.ext)
    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir)
    exclude_globs = DEFAULT_EXCLUDE_GLOBS + args.exclude_glob

    candidates: list[Path] = []
    for scan_root in args.scan_root:
        scan_root = scan_root.resolve()
        if not scan_root.exists():
            print(f"Scan root missing, skipping: {scan_root}", file=sys.stderr)
            continue
        candidates.extend(
            iter_files(scan_root, exts, exclude_dirs, exclude_globs, args.max_bytes)
        )

    pending: list[Path] = [path for path in candidates if needs_index(path, state)]
    if args.dry_run:
        print(f"Found {len(candidates)} files, {len(pending)} pending.")
        return 0

    if not pending:
        print("No new or changed files to index.")
        return 0

    batch_size = max(args.batch_size, 1)
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    max_batches = args.max_batches
    if max_batches > 0:
        batches = batches[:max_batches]

    for index, batch in enumerate(batches, start=1):
        print(f"Processing batch {index}/{len(batches)} with {len(batch)} files.")
        clear_input_dir(input_dir)
        for scan_root in args.scan_root:
            scan_root = scan_root.resolve()
            files = [path for path in batch if path.is_relative_to(scan_root)]
            if files:
                stage_files(input_dir, scan_root, files, args.copy)

        if not state.get("indexed") and not has_existing_output(root):
            cmd = ["index", "--root", str(root), "--method", args.method]
            if args.verbose:
                cmd.append("--verbose")
            run_graphrag(args.graphrag_cmd, cmd)
            state["indexed"] = True
        else:
            cmd = ["update", "--root", str(root), "--method", args.method]
            if args.verbose:
                cmd.append("--verbose")
            run_graphrag(args.graphrag_cmd, cmd)

        for path in batch:
            stat = path.stat()
            state["processed"][str(path)] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        save_state(state_path, state)

    print("Indexing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
