#!/usr/bin/env python3
"""Benchmark agentic_chess matches.

Example:
  python game_forge/tools/agentic_chess_benchmark.py --games 5 --max-moves 60 --output artifacts/agentic_chess_benchmark.json
  python game_forge/tools/agentic_chess_benchmark.py --white random --black random --games 3 --max-moves 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

script_path = Path(__file__).resolve()
repo_root = script_path.parents[2]
extra_paths = [
    repo_root,
    repo_root / "game_forge" / "src",
    repo_root / "lib",
    repo_root / "agent_forge" / "src",
]
for path in extra_paths:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from agentic_chess.agents import AgentForgeAgent, RandomAgent
from agentic_chess.engine import DependencyError, MatchConfig, play_match

AGENT_CHOICES = ["random", "agent-forge"]
AGENT_POLICIES = ["capture", "random"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark agentic_chess matches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=5, help="Number of games to run")
    parser.add_argument("--max-moves", type=int, default=80, help="Maximum plies per game")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--white", choices=AGENT_CHOICES, default="random", help="White agent")
    parser.add_argument(
        "--black",
        choices=AGENT_CHOICES,
        default="agent-forge",
        help="Black agent",
    )
    parser.add_argument(
        "--agent-forge-policy",
        choices=AGENT_POLICIES,
        default="capture",
        help="Agent Forge move policy",
    )
    parser.add_argument(
        "--agent-forge-memory",
        type=Path,
        help="Optional agent_forge memory directory",
    )
    parser.add_argument(
        "--agent-forge-no-git",
        action="store_true",
        help="Disable git-backed memory when using --agent-forge-memory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write benchmark summary to JSON",
    )
    return parser.parse_args()


def build_agent(kind: str, args: argparse.Namespace) -> Any:
    if kind == "random":
        return RandomAgent()
    if kind == "agent-forge":
        memory_dir = str(args.agent_forge_memory) if args.agent_forge_memory else None
        return AgentForgeAgent(
            policy=args.agent_forge_policy,
            memory_dir=memory_dir,
            git_enabled=not args.agent_forge_no_git,
        )
    raise ValueError(f"Unknown agent kind: {kind}")


def summarize_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter.keys())}


def summarize_moves(moves: list[int]) -> dict[str, float]:
    if not moves:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": float(min(moves)),
        "max": float(max(moves)),
        "avg": float(sum(moves)) / len(moves),
    }


def main() -> int:
    args = parse_args()
    if args.games <= 0:
        print("ERROR --games must be >= 1", file=sys.stderr)
        return 2
    if args.max_moves <= 0:
        print("ERROR --max-moves must be >= 1", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    white = build_agent(args.white, args)
    black = build_agent(args.black, args)

    results = Counter[str]()
    terminations = Counter[str]()
    move_counts: list[int] = []

    start = time.perf_counter()
    try:
        for _ in range(args.games):
            config = MatchConfig(
                max_moves=args.max_moves,
                seed=rng.randint(0, 1_000_000),
            )
            outcome = play_match(white, black, config)
            results[outcome.result] += 1
            terminations[outcome.termination] += 1
            move_counts.append(len(outcome.moves))
    except DependencyError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "games": args.games,
        "max_moves": args.max_moves,
        "seed": args.seed,
        "white": args.white,
        "black": args.black,
        "agent_forge_policy": args.agent_forge_policy,
        "results": summarize_counts(results),
        "terminations": summarize_counts(terminations),
        "moves": summarize_moves(move_counts),
        "elapsed_seconds": elapsed,
        "games_per_second": args.games / elapsed if elapsed else 0.0,
    }

    print("INFO agentic_chess benchmark")
    print(f"INFO games: {args.games}")
    print(f"INFO elapsed_seconds: {elapsed:.3f}")
    print(f"INFO results: {payload['results']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"INFO wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
