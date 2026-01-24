"""Profile indexing performance using cProfile."""

from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
from typing import Sequence

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


@eidosian()
def main(argv: Sequence[str] | None = None) -> int:
    """Run the profiling CLI."""

    args = build_parser().parse_args(argv)
    profile_index(args.source_root, args.tests_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
