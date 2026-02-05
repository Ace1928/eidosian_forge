"""Module entrypoint for chess_game prototypes."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Sequence


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _ensure_local_imports() -> None:
    package_dir = _package_dir()
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))


def _script_path(name: str) -> Path:
    return _package_dir() / f"{name}.py"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chess game prototype launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=["prototype", "prototype_old", "init"],
        default="prototype",
        help="Which chess game entrypoint to run",
    )
    parser.add_argument("--list", action="store_true", help="List available variants")
    parser.add_argument("variant_args", nargs=argparse.REMAINDER, help="Arguments for the variant")
    return parser.parse_args(argv)


def _print_list() -> None:
    print("INFO available variants:")
    print("- prototype: personality-driven chess prototype")
    print("- prototype_old: legacy prototype")
    print("- init: init.py self-tests (no GUI)")


def _run_variant(args: argparse.Namespace) -> int:
    _ensure_local_imports()
    variant_args = list(args.variant_args)
    if variant_args[:1] == ["--"]:
        variant_args = variant_args[1:]
    script = _script_path(args.variant)
    sys.argv = [str(script), *variant_args]
    globals_dict = runpy.run_path(str(script))
    main_fn = globals_dict.get("main")
    if main_fn is None:
        return 0
    return int(main_fn() or 0)


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
    return _run_variant(args)


if __name__ == "__main__":
    raise SystemExit(main())
