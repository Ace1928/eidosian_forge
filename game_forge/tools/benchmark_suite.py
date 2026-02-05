#!/usr/bin/env python3
"""Run a curated benchmark sweep for game_forge.

Example:
  python game_forge/tools/benchmark_suite.py --list
  python game_forge/tools/benchmark_suite.py --dry-run --only agentic-chess,falling-sand
  python game_forge/tools/benchmark_suite.py --preset full --output-dir game_forge/tools/artifacts
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import uuid
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Benchmark:
    name: str
    description: str
    target: str
    quick_args: Sequence[str]
    full_args: Sequence[str]
    requires: Sequence[str] = ()
    module_checks: Sequence[str] = ()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_name_list(value: str) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def default_tag() -> str:
    return dt.date.today().strftime("%Y%m%d")


def build_output_path(output_dir: Path, name: str, tag: str) -> Path:
    safe_name = name.replace(" ", "_").replace("/", "_")
    return output_dir / f"{safe_name}_{tag}.json"


def json_dump(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def missing_modules(modules: Iterable[str]) -> list[str]:
    missing = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def benchmarks() -> list[Benchmark]:
    return [
        Benchmark(
            name="agentic-chess",
            description="Agentic chess match benchmark",
            target="agentic-chess-benchmark",
            quick_args=["--games", "2", "--max-moves", "30", "--output", "{output}"],
            full_args=["--games", "10", "--max-moves", "80", "--output", "{output}"],
            requires=["python-chess"],
            module_checks=["chess"],
        ),
        Benchmark(
            name="gene-particles",
            description="Gene Particles headless benchmark",
            target="gene-particles-benchmark",
            quick_args=[
                "--steps",
                "100",
                "--cell-types",
                "4",
                "--particles",
                "200",
                "--output",
                "{output}",
            ],
            full_args=[
                "--steps",
                "500",
                "--cell-types",
                "6",
                "--particles",
                "500",
                "--output",
                "{output}",
            ],
            requires=["numpy", "pygame"],
            module_checks=["numpy", "pygame"],
        ),
        Benchmark(
            name="algorithms-lab",
            description="Algorithms Lab benchmark suite",
            target="algorithms-lab-benchmark",
            quick_args=["--algorithms", "all", "--particles", "256", "--steps", "50", "--output", "{output}"],
            full_args=["--algorithms", "all", "--particles", "512", "--steps", "100", "--output", "{output}"],
            requires=["numpy"],
            module_checks=["numpy"],
        ),
        Benchmark(
            name="falling-sand",
            description="Falling Sand benchmark suite",
            target="falling-sand-benchmark",
            quick_args=["--runs", "3", "--output", "{output}"],
            full_args=["--runs", "10", "--output", "{output}"],
            requires=["numpy"],
            module_checks=["numpy"],
        ),
        Benchmark(
            name="stratum",
            description="Stratum performance benchmark",
            target="stratum-benchmark",
            quick_args=["--grid", "16", "--ticks", "5", "--output", "{output}"],
            full_args=["--grid", "32", "--ticks", "20", "--output", "{output}"],
        ),
        Benchmark(
            name="pyparticles-engine",
            description="PyParticles engine benchmark",
            target="pyparticles-benchmark",
            quick_args=[
                "--particles",
                "256",
                "--steps",
                "5",
                "--dt",
                "0.01",
                "--warmup",
                "1",
                "--no-profile",
                "--output",
                "{output}",
            ],
            full_args=[
                "--particles",
                "512",
                "--steps",
                "10",
                "--dt",
                "0.01",
                "--warmup",
                "1",
                "--no-profile",
                "--output",
                "{output}",
            ],
            requires=["numpy"],
            module_checks=["numpy"],
        ),
        Benchmark(
            name="pyparticles-sim",
            description="PyParticles simulation benchmark",
            target="pyparticles-benchmark-sim",
            quick_args=[
                "--particles",
                "256",
                "--steps",
                "5",
                "--dt",
                "0.01",
                "--warmup",
                "1",
                "--no-profile",
                "--output",
                "{output}",
            ],
            full_args=[
                "--particles",
                "512",
                "--steps",
                "10",
                "--dt",
                "0.01",
                "--warmup",
                "1",
                "--no-profile",
                "--output",
                "{output}",
            ],
            requires=["numpy"],
            module_checks=["numpy"],
        ),
    ]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a curated benchmark sweep for game_forge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument(
        "--preset",
        choices=["quick", "full"],
        default="quick",
        help="Benchmark size preset",
    )
    parser.add_argument("--only", type=str, default="", help="Comma-separated benchmark names to run")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated benchmark names to skip")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="game_forge/tools/artifacts",
        help="Directory to write benchmark JSON outputs",
    )
    parser.add_argument("--tag", type=str, default="", help="Tag suffix for output filenames")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failures")
    parser.add_argument(
        "--summary",
        type=str,
        default="",
        help="Write aggregate summary JSON to this path",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        default=True,
        help="Skip benchmarks with missing dependencies",
    )
    parser.add_argument(
        "--no-check-deps",
        dest="check_deps",
        action="store_false",
        help="Attempt benchmarks even if dependencies are missing",
    )
    return parser.parse_args(argv)


def list_benchmarks(items: Sequence[Benchmark]) -> None:
    print("INFO available benchmarks:")
    for bench in items:
        requires = f" (requires: {', '.join(bench.requires)})" if bench.requires else ""
        print(f"- {bench.name}: {bench.description}{requires}")


def render_args(args: Sequence[str], output_path: Path) -> list[str]:
    rendered = []
    for arg in args:
        rendered.append(arg.format(output=str(output_path)))
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    items = benchmarks()

    if args.list:
        list_benchmarks(items)
        return 0

    only = parse_name_list(args.only)
    skip = parse_name_list(args.skip)
    output_dir = Path(args.output_dir)
    tag = args.tag or default_tag()

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    run_py = repo_root() / "game_forge" / "tools" / "run.py"
    run_id = str(uuid.uuid4())
    generated_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    failures = 0
    skipped = 0
    ran = 0
    records: list[dict[str, object]] = []

    for bench in items:
        if only and bench.name not in only:
            continue
        if bench.name in skip:
            continue

        if args.check_deps:
            missing = missing_modules(bench.module_checks)
            if missing:
                missing_list = ", ".join(missing)
                print(f"WARN skipping {bench.name}: missing modules {missing_list}")
                skipped += 1
                records.append(
                    {
                        "benchmark_id": bench.name,
                        "name": bench.name,
                        "target": bench.target,
                        "status": "skipped",
                        "reason": f"missing modules: {missing_list}",
                        "args": [],
                        "command": [],
                    }
                )
                continue

        output_path = build_output_path(output_dir, bench.name, tag)
        bench_args = bench.quick_args if args.preset == "quick" else bench.full_args
        rendered_args = render_args(bench_args, output_path)
        cmd = [sys.executable, str(run_py), bench.target, "--", *rendered_args]

        if args.dry_run:
            print("INFO dry-run:", bench.name)
            print("INFO command:", " ".join(cmd))
            records.append(
                {
                    "benchmark_id": bench.name,
                    "name": bench.name,
                    "target": bench.target,
                    "status": "dry-run",
                    "output": str(output_path),
                    "args": rendered_args,
                    "command": cmd,
                }
            )
            continue

        print(f"INFO running {bench.name}")
        ran += 1
        exit_code = subprocess.call(cmd)
        status = "ok" if exit_code == 0 else "failed"
        record: dict[str, object] = {
            "benchmark_id": bench.name,
            "name": bench.name,
            "target": bench.target,
            "status": status,
            "output": str(output_path),
            "exit_code": exit_code,
            "args": rendered_args,
            "command": cmd,
        }
        if exit_code != 0:
            failures += 1
            print(f"ERROR {bench.name} failed with exit code {exit_code}")
            records.append(record)
            if not args.keep_going:
                break
        else:
            records.append(record)

    if args.dry_run:
        if args.summary:
            summary_path = Path(args.summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "run_id": run_id,
                "generated_at": generated_at,
                "preset": args.preset,
                "tag": tag,
                "output_dir": str(output_dir),
                "dry_run": True,
                "ran": ran,
                "failed": failures,
                "skipped": skipped,
                "benchmarks": records,
            }
            summary_path.write_text(json_dump(summary), encoding="utf-8")
            print(f"INFO wrote summary to {summary_path}")
        return 0

    print(f"INFO benchmark suite complete: ran={ran} failed={failures} skipped={skipped}")
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_id": run_id,
            "generated_at": generated_at,
            "preset": args.preset,
            "tag": tag,
            "output_dir": str(output_dir),
            "dry_run": False,
            "ran": ran,
            "failed": failures,
            "skipped": skipped,
            "benchmarks": records,
        }
        summary_path.write_text(json_dump(summary), encoding="utf-8")
        print(f"INFO wrote summary to {summary_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
