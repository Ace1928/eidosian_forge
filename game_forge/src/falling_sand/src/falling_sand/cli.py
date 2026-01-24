"""CLI entrypoint for falling sand tooling."""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Sequence

from falling_sand import benchmarks, indexer, ingest, reporting
from falling_sand.engine.demo import run_demo
from eidosian_core import eidosian

Command = Callable[[Sequence[str] | None], int]


@eidosian()
def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser with subcommands."""

    parser = argparse.ArgumentParser(prog="falling-sand", description="Falling sand toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="Launch Panda3D demo")

    bench_parser = subparsers.add_parser("bench", help="Run benchmark suite")
    bench_parser.add_argument("--source-root")
    bench_parser.add_argument("--tests-root")
    bench_parser.add_argument("--runs")
    bench_parser.add_argument("--output")

    index_parser = subparsers.add_parser("index", help="Run indexer")
    index_parser.add_argument("--source-root")
    index_parser.add_argument("--tests-root")
    index_parser.add_argument("--output")
    index_parser.add_argument("--test-report", action="append")
    index_parser.add_argument("--profile-stats")
    index_parser.add_argument("--profile-top-n")
    index_parser.add_argument("--benchmark-report", action="append")
    index_parser.add_argument("--exclude-dir", action="append")
    index_parser.add_argument("--allow-missing-tests", action="store_true")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest index into SQLite")
    ingest_parser.add_argument("--index")
    ingest_parser.add_argument("--db")
    ingest_parser.add_argument("--batch-size")

    report_parser = subparsers.add_parser("report", help="Generate report from SQLite")
    report_parser.add_argument("--db")
    report_parser.add_argument("--output")
    report_parser.add_argument("--run-limit")
    report_parser.add_argument("--top-n")

    return parser


@eidosian()
def resolve_command(name: str) -> Command:
    """Resolve a subcommand to its handler."""

    if name not in _COMMANDS:
        raise ValueError(f"Unknown command: {name}")
    return _COMMANDS[name]


def _run_demo(argv: Sequence[str] | None = None) -> int:
    run_demo()
    return 0


_COMMANDS: dict[str, Command] = {
    "demo": _run_demo,
    "bench": benchmarks.main,
    "index": indexer.main,
    "ingest": ingest.main,
    "report": reporting.main,
}


@eidosian()
def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    command = resolve_command(args.command)

    if argv is None:
        return command(sys.argv[2:])
    return command(argv[1:])
