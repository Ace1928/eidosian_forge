#!/usr/bin/env python3
"""Moltbook Forge CLI wrapper."""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Iterable

from moltbook_forge.moltbook_bootstrap import main as bootstrap_main
from moltbook_forge.moltbook_ingest import main as ingest_main
from moltbook_forge.moltbook_quarantine import main as quarantine_main
from moltbook_forge.moltbook_sanitize import main as sanitize_main
from moltbook_forge.moltbook_screen import main as screen_main
from moltbook_forge.moltbook_skill_review import main as skill_review_main
from moltbook_forge.moltbook_validate import main as validate_main


CommandMain = Callable[[Iterable[str]], int]

COMMANDS: dict[str, CommandMain] = {
    "sanitize": sanitize_main,
    "screen": screen_main,
    "validate": validate_main,
    "quarantine": quarantine_main,
    "skill-review": skill_review_main,
    "ingest": ingest_main,
    "bootstrap": bootstrap_main,
}


def parse_args(argv: Iterable[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Moltbook Forge CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List available commands")
    subparsers = parser.add_subparsers(dest="command")
    for name in sorted(COMMANDS.keys()):
        subparsers.add_parser(name, help=f"Run {name}")
    return parser.parse_known_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args, unknown = parse_args(argv or sys.argv[1:])
    if args.list or not args.command:
        print("INFO available commands:")
        for name in sorted(COMMANDS.keys()):
            print(f"- {name}")
        return 0 if args.list else 2

    command_main = COMMANDS[args.command]
    forwarded = list(unknown)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    return command_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
