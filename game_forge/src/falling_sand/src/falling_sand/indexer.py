"""Project indexer for source and test code."""

from __future__ import annotations

import argparse
import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from falling_sand.models import BenchmarkSummary, IndexDocument, IndexEntry, Origin
from falling_sand.reports import read_benchmark_report, read_junit_reports, read_profile_stats
from falling_sand.schema import CURRENT_SCHEMA_VERSION


DEFAULT_EXCLUDE_DIRS = (".git", "__pycache__", ".venv", "venv", "artifacts")


def normalize_exclude_dirs(exclude_dirs: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize exclude directories into a unique tuple."""

    if not exclude_dirs:
        return DEFAULT_EXCLUDE_DIRS
    seen: set[str] = set()
    normalized: list[str] = []
    for value in exclude_dirs:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def validate_root(root: Path, allow_missing: bool) -> None:
    """Validate that a root exists and is a directory."""

    if not root.exists():
        if allow_missing:
            return
        raise ValueError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Root is not a directory: {root}")


def should_exclude(path: Path, exclude_set: set[str]) -> bool:
    """Return True if a path should be excluded."""

    return any(part in exclude_set for part in path.parts)


def iter_python_files(root: Path, exclude_dirs: Sequence[str]) -> Iterator[Path]:
    """Yield Python files under root, excluding specified directories."""

    validate_root(root, allow_missing=False)
    exclude_set = set(exclude_dirs)

    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name in exclude_set:
                            continue
                        stack.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if not entry.name.endswith(".py"):
                        continue
                    if exclude_set and should_exclude(Path(entry.path), exclude_set):
                        continue
                    yield Path(entry.path)
        except OSError as exc:
            raise ValueError(f"Unable to read directory {current}: {exc}") from exc


def parse_python_file(path: Path) -> ast.Module:
    """Parse a Python source file into an AST."""

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read {path}: {exc}") from exc

    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in {path}: {exc}") from exc


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Build a readable signature string from a function node."""

    try:
        return f"({ast.unparse(node.args)})"
    except Exception:
        return None


def module_name_from_path(path: Path, package_root: Path) -> str:
    """Derive a module name from a file path within a package root."""

    relative = path.relative_to(package_root)
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return "__init__"
    return ".".join(parts)


def extract_definitions(
    tree: ast.Module,
    module: str,
    filepath: str,
    origin: Origin,
) -> list[IndexEntry]:
    """Extract classes, functions, and methods from a module AST."""

    entries: list[IndexEntry] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entries.append(
                IndexEntry(
                    name=node.name,
                    qualname=node.name,
                    kind="function",
                    origin=origin,
                    module=module,
                    filepath=filepath,
                    lineno=node.lineno,
                    docstring=ast.get_docstring(node),
                    signature=format_signature(node),
                )
            )
        elif isinstance(node, ast.ClassDef):
            entries.append(
                IndexEntry(
                    name=node.name,
                    qualname=node.name,
                    kind="class",
                    origin=origin,
                    module=module,
                    filepath=filepath,
                    lineno=node.lineno,
                    docstring=ast.get_docstring(node),
                    signature=None,
                )
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entries.append(
                        IndexEntry(
                            name=child.name,
                            qualname=f"{node.name}.{child.name}",
                            kind="method",
                            origin=origin,
                            module=module,
                            filepath=filepath,
                            lineno=child.lineno,
                            docstring=ast.get_docstring(child),
                            signature=format_signature(child),
                        )
                    )
    return entries


def index_root(root: Path, origin: Origin, exclude_dirs: Sequence[str]) -> list[IndexEntry]:
    """Index all Python files rooted at the provided directory."""

    if not root.exists():
        return []
    entries: list[IndexEntry] = []
    for path in iter_python_files(root, exclude_dirs=exclude_dirs):
        module = module_name_from_path(path, root)
        tree = parse_python_file(path)
        entries.extend(extract_definitions(tree, module, str(path), origin))
    return entries


def index_project(
    source_root: Path,
    tests_root: Path,
    exclude_dirs: Sequence[str] | None = None,
    test_report_paths: Sequence[Path] | None = None,
    profile_stats_path: Path | None = None,
    benchmark_report_path: Sequence[Path] | None = None,
    profile_top_n: int = 20,
    allow_missing_tests: bool = False,
) -> IndexDocument:
    """Index source and test code and return a structured document."""

    excluded = normalize_exclude_dirs(exclude_dirs)
    validate_root(source_root, allow_missing=False)
    validate_root(tests_root, allow_missing=allow_missing_tests)

    source_entries = index_root(source_root, "source", excluded)
    test_entries = [] if allow_missing_tests and not tests_root.exists() else index_root(tests_root, "test", excluded)
    all_entries = tuple(sorted(source_entries + test_entries, key=lambda e: (e.module, e.lineno)))

    stats = {
        "total": len(all_entries),
        "source": len(source_entries),
        "test": len(test_entries),
        "functions": sum(1 for e in all_entries if e.kind == "function"),
        "classes": sum(1 for e in all_entries if e.kind == "class"),
        "methods": sum(1 for e in all_entries if e.kind == "method"),
    }

    test_summary = None
    if test_report_paths:
        test_summary = read_junit_reports(test_report_paths)

    profile_summary = None
    if profile_stats_path:
        profile_summary = read_profile_stats(profile_stats_path, top_n=profile_top_n)

    benchmark_summary = None
    if benchmark_report_path:
        summaries = [read_benchmark_report(path) for path in benchmark_report_path]
        cases = tuple(case for summary in summaries for case in summary.cases)
        benchmark_summary = BenchmarkSummary(cases=cases) if cases else None

    return IndexDocument(
        schema_version=CURRENT_SCHEMA_VERSION,
        entries=all_entries,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_root=str(source_root),
        tests_root=str(tests_root),
        stats=stats,
        test_summary=test_summary,
        profile_summary=profile_summary,
        benchmark_summary=benchmark_summary,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Index project source and tests.")
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exclude-dir", action="append", default=[])
    parser.add_argument("--allow-missing-tests", action="store_true")
    parser.add_argument("--test-report", type=Path, action="append", default=[])
    parser.add_argument("--profile-stats", type=Path)
    parser.add_argument("--profile-top-n", type=int, default=20)
    parser.add_argument("--benchmark-report", type=Path, action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the indexer CLI."""

    args = build_parser().parse_args(argv)
    result = index_project(
        args.source_root,
        args.tests_root,
        exclude_dirs=tuple(args.exclude_dir),
        test_report_paths=tuple(args.test_report),
        profile_stats_path=args.profile_stats,
        benchmark_report_path=tuple(args.benchmark_report),
        profile_top_n=args.profile_top_n,
        allow_missing_tests=args.allow_missing_tests,
    )
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
