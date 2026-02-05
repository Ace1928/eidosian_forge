"""Profile indexing performance using cProfile."""

from __future__ import annotations

import argparse
import cProfile
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ is None:
    sys.path.append(str(PROJECT_ROOT / "src"))

from falling_sand.indexer import index_project
from eidosian_core import eidosian


@eidosian()
def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for profiling."""

    parser = argparse.ArgumentParser(description="Profile project indexing.")
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/profile.pstats"))
    return parser


@eidosian()
def profile_index(
    source_root: Path,
    tests_root: Path,
    output: Path,
) -> Path:
    """Profile the indexer and write profiler stats to output."""

    output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.runcall(index_project, source_root, tests_root)
    profiler.dump_stats(str(output))
    return output


def _resolve_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


@eidosian()
def main(argv: Sequence[str] | None = None) -> int:
    """Run the profiling CLI."""

    args = build_parser().parse_args(argv)
    source_root = _resolve_root(args.source_root)
    tests_root = _resolve_root(args.tests_root)
    profile_index(source_root, tests_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
