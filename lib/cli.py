"""Shared CLI helpers for Eidosian entrypoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, NamedTuple, Type


class Colors:
    """ANSI color palette with opt-out support."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"

    _enabled = bool(sys.stdout.isatty()) and "NO_COLOR" not in os.environ
    _original = {
        "RESET": RESET,
        "BOLD": BOLD,
        "RED": RED,
        "GREEN": GREEN,
        "YELLOW": YELLOW,
        "CYAN": CYAN,
    }

    @classmethod
    def disable(cls) -> None:
        cls._enabled = False
        for key in cls._original:
            setattr(cls, key, "")

    @classmethod
    def enable(cls) -> None:
        cls._enabled = True
        for key, value in cls._original.items():
            setattr(cls, key, value)


class CommandResult(NamedTuple):
    success: bool
    message: str = ""
    data: Any = None


class ForgeDetector:
    """Best-effort module availability checker."""

    @staticmethod
    def is_available(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False


class StandardCLI:
    """Base class for forge CLIs."""

    name = "forge"
    description = "Forge CLI"
    version = "0.1.0"

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            prog=self.name.replace("_", "-"),
            description=self.description,
        )
        self.parser.add_argument("--json", action="store_true", help="Output JSON")
        self.parser.add_argument("--no-color", action="store_true", help="Disable colors")
        self.parser.add_argument("-q", "--quiet", action="store_true", help="Suppress informational output")
        self.parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"%(prog)s {self.version}",
        )

        subparsers = self.parser.add_subparsers(dest="command")
        status_parser = subparsers.add_parser("status", help="Show forge status")
        status_parser.set_defaults(func=self._cmd_status)
        self.register_commands(subparsers)

    def register_commands(self, subparsers: argparse._SubParsersAction) -> None:
        """Override in subclasses to register forge-specific commands."""
        raise NotImplementedError

    def cmd_status(self, args: argparse.Namespace) -> CommandResult:
        """Override in subclasses for forge-specific status."""
        return CommandResult(True, f"{self.name} is operational")

    def get_enhanced_capabilities(self, available_forges: list[str]) -> list[str]:
        """Optional integration hints for other loaded forges."""
        return []

    def _cmd_status(self, args: argparse.Namespace) -> int:
        result = self.cmd_status(args)
        self._output(result, args)
        return 0 if result.success else 1

    def _output(self, result: CommandResult, args: argparse.Namespace) -> None:
        """Render a command result in text or JSON."""
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "success": result.success,
                        "message": result.message,
                        "data": result.data,
                    },
                    indent=2,
                    default=str,
                )
            )
            return

        if result.message and not getattr(args, "quiet", False):
            color = Colors.GREEN if result.success else Colors.RED
            print(f"{color}{result.message}{Colors.RESET}")

        if result.data is None:
            return

        if isinstance(result.data, (dict, list)):
            print(json.dumps(result.data, indent=2, default=str))
        else:
            print(result.data)

    def run(self, argv: list[str] | None = None) -> int:
        args = self.parser.parse_args(argv)

        if getattr(args, "no_color", False):
            Colors.disable()

        if not hasattr(args, "func"):
            self.parser.print_help()
            return 0

        try:
            outcome = args.func(args)
            if isinstance(outcome, int):
                return outcome
            if isinstance(outcome, CommandResult):
                self._output(outcome, args)
                return 0 if outcome.success else 1
            return 0
        except KeyboardInterrupt:
            if not getattr(args, "quiet", False):
                print(f"{Colors.YELLOW}Interrupted{Colors.RESET}")
            return 130
        except Exception as exc:
            if getattr(args, "json", False):
                print(json.dumps({"success": False, "error": str(exc)}, indent=2))
            else:
                print(f"{Colors.RED}Error:{Colors.RESET} {exc}")
            return 1


def create_cli_entry_point(cli_class: Type[StandardCLI]):
    """Create a conventional `main()` entry point for a forge CLI class."""

    def _entry() -> None:
        cli = cli_class()
        raise SystemExit(cli.run())

    return _entry
