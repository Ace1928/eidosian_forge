"""Module entrypoint for Algorithms Lab."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _demo_path() -> Path:
    return _repo_root() / "game_forge" / "tools" / "algorithms_lab" / "demo.py"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algorithms Lab launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List available actions")
    parser.add_argument("--demo", action="store_true", help="Run the Algorithms Lab demo")
    parser.add_argument("demo_args", nargs=argparse.REMAINDER, help="Arguments for the demo")
    return parser.parse_args(argv)


def _print_list() -> None:
    print("INFO available actions:")
    print("- demo: run the algorithms lab demo (use --demo -- <args>)")


def _run_demo(args: argparse.Namespace) -> int:
    demo_path = _demo_path()
    demo_args = list(args.demo_args)
    if demo_args[:1] == ["--"]:
        demo_args = demo_args[1:]
    sys.argv = [str(demo_path), *demo_args]
    globals_dict = runpy.run_path(str(demo_path))
    demo_main = globals_dict.get("main")
    if demo_main is None:
        raise SystemExit("ERROR demo main() not found")
    return int(demo_main() or 0)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        _print_list()
        return 0
    args = parse_args(argv)
    if args.list:
        _print_list()
        return 0
    if args.demo or args.demo_args:
        return _run_demo(args)
    _print_list()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
