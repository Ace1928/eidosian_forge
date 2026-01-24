"""Benchmark the indexing workflow."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Sequence

from falling_sand.indexer import index_project
from eidosian_core import eidosian


@eidosian()
def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for benchmarking."""

    parser = argparse.ArgumentParser(description="Benchmark project indexing.")
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    return parser


@eidosian()
def benchmark_index(source_root: Path, tests_root: Path, runs: int) -> list[float]:
    """Benchmark the indexer and return run durations in seconds."""

    if runs <= 0:
        raise ValueError("runs must be positive")

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        index_project(source_root, tests_root)
        samples.append(time.perf_counter() - start)
    return samples


@eidosian()
def write_benchmark_report(samples: list[float], output: Path) -> None:
    """Write benchmark stats to a JSON report."""

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": len(samples),
        "samples": samples,
        "mean_seconds": statistics.mean(samples),
        "median_seconds": statistics.median(samples),
        "stdev_seconds": statistics.pstdev(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@eidosian()
def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""

    args = build_parser().parse_args(argv)
    samples = benchmark_index(args.source_root, args.tests_root, args.runs)
    write_benchmark_report(samples, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
