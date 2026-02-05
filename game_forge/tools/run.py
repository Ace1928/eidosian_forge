#!/usr/bin/env python3
"""Launcher for game_forge simulations.

Example:
  python game_forge/tools/run.py --list
  python game_forge/tools/run.py gene-particles
  python game_forge/tools/run.py ecosmos -- --ticks 100
  python game_forge/tools/run.py algorithms-lab-demo -- --algorithm sph --visual
  python game_forge/tools/run.py stratum-benchmark -- --grid 16 --ticks 10
  python game_forge/tools/run.py agentic-chess -- --white random --black agent-forge
  python game_forge/tools/run.py agentic-chess-benchmark -- --games 5 --max-moves 60
  python game_forge/tools/run.py falling-sand-benchmark -- --runs 5
  python game_forge/tools/run.py pyparticles-benchmark -- --particles 512 --steps 10
  python game_forge/tools/run.py pyparticles-benchmark-sim -- --particles 512 --steps 10
  python game_forge/tools/run.py algorithms-lab-profiler -- --algorithm grid --steps 10 --output artifacts/algorithms_lab.prof
  python game_forge/tools/run.py falling-sand-profile-index -- --runs 1 --output artifacts/profile.pstats
  python game_forge/tools/run.py falling-sand-index -- --output artifacts/index.json
  python game_forge/tools/run.py falling-sand-ingest -- --index artifacts/index.json --db artifacts/index.db
  python game_forge/tools/run.py falling-sand-report -- --db artifacts/index.db --output artifacts/report.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Target:
    name: str
    description: str
    command: Sequence[str]
    extra_pythonpath: Sequence[Path] = ()
    requires: Sequence[str] = ()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def unique_paths(paths: Iterable[Path | str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in paths:
        value = str(item)
        if value not in seen:
            seen[value] = None
    return list(seen.keys())


def build_pythonpath(root: Path, extra: Iterable[Path]) -> str:
    parts: list[Path | str] = [
        root,
        root / "lib",
        root / "game_forge" / "src",
    ]
    parts.extend(extra)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(unique_paths(parts))


def load_targets(root: Path) -> dict[str, Target]:
    return {
        "algorithms-lab": Target(
            name="algorithms-lab",
            description="Algorithms Lab demo via module entrypoint",
            command=["-m", "algorithms_lab", "--demo"],
            requires=["numpy", "pygame (visual)"],
        ),
        "agentic-chess": Target(
            name="agentic-chess",
            description="Agentic chess match runner",
            command=["-m", "agentic_chess"],
            extra_pythonpath=[root / "agent_forge" / "src"],
            requires=["python-chess"],
        ),
        "agentic-chess-benchmark": Target(
            name="agentic-chess-benchmark",
            description="Agentic chess benchmark runner",
            command=[str(root / "game_forge" / "tools" / "agentic_chess_benchmark.py")],
            extra_pythonpath=[root / "agent_forge" / "src"],
            requires=["python-chess"],
        ),
        "chess-game-prototype": Target(
            name="chess-game-prototype",
            description="Chess game personality-driven prototype",
            command=[str(root / "game_forge" / "src" / "chess_game" / "prototype.py")],
            extra_pythonpath=[root / "game_forge" / "src" / "chess_game"],
            requires=["pygame", "python-chess", "requests (optional)"],
        ),
        "chess-game-prototype-old": Target(
            name="chess-game-prototype-old",
            description="Legacy chess game prototype",
            command=[str(root / "game_forge" / "src" / "chess_game" / "prototype_old.py")],
            extra_pythonpath=[root / "game_forge" / "src" / "chess_game"],
            requires=["pygame", "python-chess", "requests (optional)"],
        ),
        "ecosmos": Target(
            name="ecosmos",
            description="ECosmos evolving ecosystem simulation",
            command=[str(root / "game_forge" / "src" / "ECosmos" / "main.py")],
            extra_pythonpath=[root / "game_forge" / "src" / "ECosmos"],
            requires=["numpy", "matplotlib (visual)"],
        ),
        "gene-particles": Target(
            name="gene-particles",
            description="Gene Particles simulation",
            command=[str(root / "game_forge" / "src" / "gene_particles" / "gp_main.py")],
            requires=["numpy", "pygame"],
        ),
        "gene-particles-benchmark": Target(
            name="gene-particles-benchmark",
            description="Gene Particles headless benchmark",
            command=[str(root / "game_forge" / "tools" / "gene_particles_benchmark.py")],
            requires=["numpy", "pygame"],
        ),
        "gene-particles-profile": Target(
            name="gene-particles-profile",
            description="Gene Particles headless profile run",
            command=[str(root / "game_forge" / "tools" / "gene_particles_profile.py")],
            requires=["numpy", "pygame"],
        ),
        "algorithms-lab-benchmark": Target(
            name="algorithms-lab-benchmark",
            description="Algorithms Lab benchmark suite",
            command=[str(root / "game_forge" / "tools" / "algorithms_lab" / "benchmark.py")],
            requires=["numpy"],
        ),
        "algorithms-lab-profiler": Target(
            name="algorithms-lab-profiler",
            description="Algorithms Lab cProfile run",
            command=[str(root / "game_forge" / "tools" / "algorithms_lab" / "profiler.py")],
            requires=["numpy"],
        ),
        "algorithms-lab-demo": Target(
            name="algorithms-lab-demo",
            description="Algorithms Lab demo runner",
            command=[str(root / "game_forge" / "tools" / "algorithms_lab" / "demo.py")],
            requires=["numpy", "pygame (visual)"],
        ),
        "eidosian-universe": Target(
            name="eidosian-universe",
            description="Eidosian Universe simulation",
            command=[str(root / "game_forge" / "src" / "eidosian_universe" / "eu_main.py")],
            requires=["numpy", "pygame"],
        ),
        "falling-sand": Target(
            name="falling-sand",
            description="Falling Sand CLI",
            command=["-m", "falling_sand"],
            extra_pythonpath=[root / "game_forge" / "src" / "falling_sand" / "src"],
        ),
        "falling-sand-benchmark": Target(
            name="falling-sand-benchmark",
            description="Falling Sand benchmark suite",
            command=["-m", "falling_sand.benchmarks"],
            extra_pythonpath=[root / "game_forge" / "src" / "falling_sand" / "src"],
            requires=["numpy"],
        ),
        "falling-sand-profile-index": Target(
            name="falling-sand-profile-index",
            description="Falling Sand indexer profile",
            command=[str(root / "game_forge" / "src" / "falling_sand" / "scripts" / "profile_index.py")],
            extra_pythonpath=[root / "game_forge" / "src" / "falling_sand" / "src"],
            requires=["numpy"],
        ),
        "falling-sand-index": Target(
            name="falling-sand-index",
            description="Falling Sand indexer CLI",
            command=["-m", "falling_sand.indexer"],
            extra_pythonpath=[root / "game_forge" / "src" / "falling_sand" / "src"],
            requires=["numpy"],
        ),
        "falling-sand-ingest": Target(
            name="falling-sand-ingest",
            description="Falling Sand ingest CLI",
            command=["-m", "falling_sand.ingest"],
            extra_pythonpath=[root / "game_forge" / "src" / "falling_sand" / "src"],
            requires=["numpy"],
        ),
        "falling-sand-report": Target(
            name="falling-sand-report",
            description="Falling Sand report CLI",
            command=["-m", "falling_sand.reporting"],
            extra_pythonpath=[root / "game_forge" / "src" / "falling_sand" / "src"],
            requires=["numpy"],
        ),
        "stratum": Target(
            name="stratum",
            description="Stratum unified scenario CLI",
            command=[str(root / "game_forge" / "src" / "Stratum" / "scenarios" / "cli.py")],
        ),
        "stratum-cli": Target(
            name="stratum-cli",
            description="Stratum scenario CLI",
            command=[str(root / "game_forge" / "src" / "Stratum" / "scenarios" / "cli.py")],
        ),
        "stratum-benchmark": Target(
            name="stratum-benchmark",
            description="Stratum benchmark suite",
            command=[str(root / "game_forge" / "src" / "Stratum" / "tests" / "benchmark.py")],
        ),
        "pyparticles-benchmark": Target(
            name="pyparticles-benchmark",
            description="PyParticles engine benchmark",
            command=[str(root / "game_forge" / "pyparticles" / "benchmarks" / "benchmark.py")],
            extra_pythonpath=[root / "game_forge" / "pyparticles" / "src"],
            requires=["numpy"],
        ),
        "pyparticles-benchmark-sim": Target(
            name="pyparticles-benchmark-sim",
            description="PyParticles simulation benchmark",
            command=[str(root / "game_forge" / "pyparticles" / "benchmarks" / "benchmark_sim.py")],
            extra_pythonpath=[root / "game_forge" / "pyparticles" / "src"],
            requires=["numpy"],
        ),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch game_forge simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", nargs="?", help="Target to run (use --list to see options)")
    parser.add_argument("target_args", nargs=argparse.REMAINDER, help="Arguments for the target")
    parser.add_argument("--list", action="store_true", help="List available targets")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    return parser.parse_args(argv)


def list_targets(targets: dict[str, Target]) -> None:
    print("INFO available targets:")
    for name in sorted(targets):
        target = targets[name]
        requires = f" (requires: {', '.join(target.requires)})" if target.requires else ""
        print(f"- {name}: {target.description}{requires}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = repo_root()
    targets = load_targets(root)

    if args.list or not args.target:
        list_targets(targets)
        return 0 if args.list else 2

    if args.target not in targets:
        print(f"ERROR unknown target: {args.target}")
        list_targets(targets)
        return 2

    target = targets[args.target]
    target_args = list(args.target_args)
    if target_args[:1] == ["--"]:
        target_args = target_args[1:]

    env = dict(os.environ)
    env["PYTHONPATH"] = build_pythonpath(root, target.extra_pythonpath)
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    cmd = [sys.executable, *target.command, *target_args]

    if args.dry_run:
        print("INFO dry-run")
        print("INFO python:", sys.executable)
        print("INFO command:", " ".join(cmd))
        print("INFO PYTHONPATH:", env["PYTHONPATH"])
        return 0

    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
