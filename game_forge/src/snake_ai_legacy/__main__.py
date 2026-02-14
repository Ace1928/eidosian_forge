"""Module entrypoint for the legacy INDEGO Snake AI import."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Sequence


_VARIANT_TO_SCRIPT = {
    "classic": "Main.py",
    "standalone": "standalonesnake.py",
    "supersnake": "SuperSnake.py",
}


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _ensure_local_imports() -> None:
    package_dir = _package_dir()
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))


def _script_path(variant: str) -> Path:
    return _package_dir() / _VARIANT_TO_SCRIPT[variant]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch legacy INDEGO Snake AI variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=sorted(_VARIANT_TO_SCRIPT),
        default="classic",
        help="Legacy script variant to run",
    )
    parser.add_argument("--list", action="store_true", help="List available variants")
    parser.add_argument("--dry-run", action="store_true", help="Print selected script and exit")
    parser.add_argument(
        "variant_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected legacy script",
    )
    return parser.parse_args(argv)


def _print_list() -> None:
    print("INFO available variants:")
    print("- classic: menu-driven snake AI (BFS/DFS/A*/GA)")
    print("- standalone: standalone experimental snake sandbox")
    print("- supersnake: advanced supersnake prototype")
    print("INFO pass script arguments as: --variant <name> -- <args>")


def _run_variant(args: argparse.Namespace) -> int:
    variant_args = list(args.variant_args)
    if variant_args[:1] == ["--"]:
        variant_args = variant_args[1:]

    _ensure_local_imports()
    script = _script_path(args.variant)

    if args.dry_run:
        print(f"INFO dry-run variant={args.variant}")
        print(f"INFO script={script}")
        if variant_args:
            print("INFO args:", " ".join(variant_args))
        return 0

    old_argv = list(sys.argv)
    old_cwd = Path.cwd()
    try:
        # Legacy scripts resolve assets by relative paths from their folder.
        os.chdir(script.parent)
        sys.argv = [str(script), *variant_args]
        runpy.run_path(str(script), run_name="__main__")
        return 0
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    if args.list:
        _print_list()
        return 0
    return _run_variant(args)


if __name__ == "__main__":
    raise SystemExit(main())

