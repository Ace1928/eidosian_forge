from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .agents import AgentForgeAgent, RandomAgent
from .engine import DependencyError, MatchConfig, play_match

AGENT_CHOICES = ["random", "agent-forge"]
AGENT_POLICIES = ["capture", "random"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an agentic chess match",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list-agents", action="store_true", help="List available agents")
    parser.add_argument(
        "--list",
        dest="list_agents",
        action="store_true",
        help="Alias for --list-agents",
    )
    parser.add_argument("--white", choices=AGENT_CHOICES, default="random", help="White agent")
    parser.add_argument("--black", choices=AGENT_CHOICES, default="agent-forge", help="Black agent")
    parser.add_argument("--max-moves", type=int, default=200, help="Maximum plies to play")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log-moves", action="store_true", help="Log each move")
    parser.add_argument("--pgn-output", type=Path, help="Write PGN output to a file")
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
    return parser.parse_args(argv)


def list_agents() -> None:
    print("INFO available agents:")
    print("- random: selects random legal moves")
    print("- agent-forge: capture-first policy with optional agent_forge memory logging")


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.list_agents:
        list_agents()
        return 0

    white = build_agent(args.white, args)
    black = build_agent(args.black, args)

    config = MatchConfig(
        max_moves=args.max_moves,
        seed=args.seed,
        log_moves=args.log_moves,
        emit_pgn=bool(args.pgn_output),
    )

    try:
        result = play_match(white, black, config)
    except DependencyError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR {exc}", file=sys.stderr)
        return 1

    print(f"INFO result: {result.result}")
    print(f"INFO termination: {result.termination}")
    print(f"INFO moves: {len(result.moves)}")

    if args.pgn_output and result.pgn is not None:
        args.pgn_output.parent.mkdir(parents=True, exist_ok=True)
        args.pgn_output.write_text(result.pgn, encoding="utf-8")
        print(f"INFO wrote PGN: {args.pgn_output}")

    return 0
