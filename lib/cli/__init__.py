"""
Eidosian CLI Framework - Base classes and utilities for forge CLIs.

Provides:
- StandardCLI: Base class for all forge CLIs
- Common argument patterns
- Output formatting
- Bash completion generation
- Cross-forge integration detection

Every forge can operate standalone. When other forges are present,
enhanced capabilities are automatically enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output."""
        cls.RESET = cls.BOLD = cls.RED = cls.GREEN = ""
        cls.YELLOW = cls.BLUE = cls.MAGENTA = cls.CYAN = ""


@dataclass
class CommandResult:
    """Result of a CLI command execution."""

    success: bool
    message: str
    data: Optional[Any] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "success": self.success,
                "message": self.message,
                "data": self.data,
            },
            indent=2,
            default=str,
        )


class ForgeDetector:
    """Detects which forges are available for enhanced integration."""

    _cache: Dict[str, bool] = {}

    @classmethod
    def is_available(cls, forge_name: str) -> bool:
        """Check if a forge is importable."""
        if forge_name not in cls._cache:
            try:
                import importlib.util

                cls._cache[forge_name] = importlib.util.find_spec(forge_name) is not None
            except (ImportError, AttributeError, ValueError):
                cls._cache[forge_name] = False
        return cls._cache[forge_name]

    @classmethod
    def available_forges(cls) -> List[str]:
        """List all available forges."""
        forge_names = [
            "memory_forge",
            "knowledge_forge",
            "llm_forge",
            "code_forge",
            "word_forge",
            "crawl_forge",
            "doc_forge",
            "audit_forge",
            "refactor_forge",
            "glyph_forge",
            "figlet_forge",
            "agent_forge",
        ]
        return [f for f in forge_names if cls.is_available(f)]

    @classmethod
    def report(cls) -> str:
        """Generate availability report."""
        lines = ["Available Forges:"]
        for forge in cls.available_forges():
            lines.append(f"  ✓ {forge}")
        return "\n".join(lines)


class StandardCLI(ABC):
    """
    Base class for all forge CLIs.

    Provides standard argument parsing, output formatting,
    and integration detection.

    Usage:
        class MemoryForgeCLI(StandardCLI):
            name = "memory_forge"
            description = "Tiered memory system for EIDOS"
            version = "1.0.0"

            def register_commands(self, subparsers):
                # Add forge-specific commands
                ...

            def cmd_status(self, args):
                # Implement status command
                ...
    """

    # Override in subclasses
    name: str = "forge"
    description: str = "A forge CLI"
    version: str = "0.1.0"

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog=self.name.replace("_", "-"),
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._setup_global_args()
        self.subparsers = self.parser.add_subparsers(
            dest="command",
            title="commands",
            metavar="<command>",
        )
        self._register_standard_commands()
        self.register_commands(self.subparsers)

    def _setup_global_args(self) -> None:
        """Add global arguments available to all commands."""
        self.parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"%(prog)s {self.version}",
        )
        self.parser.add_argument(
            "--json",
            action="store_true",
            help="Output in JSON format",
        )
        self.parser.add_argument(
            "--no-color",
            action="store_true",
            help="Disable colored output",
        )
        self.parser.add_argument(
            "--quiet",
            "-q",
            action="store_true",
            help="Suppress non-essential output",
        )

    def _register_standard_commands(self) -> None:
        """Register commands available in all forge CLIs."""
        # Status command
        status_parser = self.subparsers.add_parser(
            "status",
            help="Show forge status and health",
        )
        status_parser.set_defaults(func=self._cmd_status_wrapper)

        # Info command
        info_parser = self.subparsers.add_parser(
            "info",
            help="Show forge information and capabilities",
        )
        info_parser.set_defaults(func=self._cmd_info)

        # Integrations command
        int_parser = self.subparsers.add_parser(
            "integrations",
            help="Show available forge integrations",
        )
        int_parser.set_defaults(func=self._cmd_integrations)

    @abstractmethod
    def register_commands(self, subparsers) -> None:
        """Register forge-specific commands. Override in subclasses."""
        pass

    @abstractmethod
    def cmd_status(self, args) -> CommandResult:
        """Return forge status. Override in subclasses."""
        pass

    def _cmd_status_wrapper(self, args) -> None:
        """Wrapper for status command."""
        result = self.cmd_status(args)
        self._output(result, args)

    def _cmd_info(self, args) -> None:
        """Show forge information."""
        info = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "commands": list(self.subparsers.choices.keys()),
        }
        result = CommandResult(True, f"{self.name} v{self.version}", info)
        self._output(result, args)

    def _cmd_integrations(self, args) -> None:
        """Show available integrations."""
        available = ForgeDetector.available_forges()
        enhanced = self.get_enhanced_capabilities(available)

        result = CommandResult(
            True,
            f"Found {len(available)} forges",
            {"available_forges": available, "enhanced_capabilities": enhanced},
        )
        self._output(result, args)

    def get_enhanced_capabilities(self, available_forges: List[str]) -> List[str]:
        """
        Return list of enhanced capabilities when other forges present.
        Override in subclasses to specify integration benefits.
        """
        return []

    def _output(self, result: CommandResult, args) -> None:
        """Format and print command result."""
        if getattr(args, "json", False):
            print(result.to_json())
        else:
            if getattr(args, "no_color", False):
                Colors.disable()

            if not getattr(args, "quiet", False):
                color = Colors.GREEN if result.success else Colors.RED
                print(f"{color}{result.message}{Colors.RESET}")

            if result.data:
                if isinstance(result.data, dict):
                    for key, value in result.data.items():
                        if isinstance(value, list):
                            print(f"\n{Colors.BOLD}{key}:{Colors.RESET}")
                            for item in value:
                                print(f"  • {item}")
                        else:
                            print(f"{key}: {value}")
                elif isinstance(result.data, list):
                    for item in result.data:
                        print(f"  • {item}")
                else:
                    print(result.data)

    def run(self, argv: Optional[List[str]] = None) -> int:
        """Parse arguments and execute command."""
        args = self.parser.parse_args(argv)

        if args.no_color:
            Colors.disable()

        if not args.command:
            self.parser.print_help()
            return 0

        if hasattr(args, "func"):
            try:
                args.func(args)
                return 0
            except Exception as e:
                if not args.quiet:
                    print(f"{Colors.RED}Error: {e}{Colors.RESET}", file=sys.stderr)
                return 1
        else:
            self.parser.print_help()
            return 0

    def generate_completion(self) -> str:
        """Generate bash completion script for this CLI."""
        prog = self.name.replace("_", "-")
        commands = list(self.subparsers.choices.keys())

        script = f"""# Bash completion for {prog}
# Add to ~/.bashrc: source <(path/to/{prog} --completion)

_{prog.replace("-", "_")}_completions() {{
    local cur prev commands
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    
    commands="{' '.join(commands)}"
    
    case "${{COMP_CWORD}}" in
        1)
            COMPREPLY=($(compgen -W "$commands --help --version --json --no-color --quiet" -- "$cur"))
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}}

complete -F _{prog.replace("-", "_")}_completions {prog}
"""
        return script


def create_cli_entry_point(cli_class: type) -> Callable:
    """Create a main() function for a CLI class."""

    def main():
        cli = cli_class()
        sys.exit(cli.run())

    return main
