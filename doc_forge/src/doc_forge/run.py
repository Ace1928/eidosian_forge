#!/usr/bin/env python3
from __future__ import annotations

import sys
import logging
import argparse

from .doc_forge import create_parser as create_doc_forge_parser
from .doc_forge import main as doc_forge_main
# Fixed import by using relative import and proper path structure
# Importing from package root is problematic during development
try:
    # First try relative import (when installed as package)
    from ..tests.test_command import add_test_subparsers
except (ImportError, ValueError):
    # Fall back to absolute import (when running from source)
     from tests.test_command import add_test_subparsers # type: ignore[import]

from .version import get_version_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("doc_forge")

def main() -> int:
    """
    Orchestrates argument parsing, sets debug flags, shows version info if requested,
    and routes commands to their handlers. Returns an integer exit code for the CLI.
    """
    parser: argparse.ArgumentParser = create_main_parser()
    args: argparse.Namespace = parser.parse_args()

    if getattr(args, "debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode activated.")

    if getattr(args, "version", False):
        print(f"Doc Forge v{get_version_string()}")
        return 0

    cmd_type: str | None = getattr(args, "command_type", None)
    if cmd_type == "docs":
        return doc_forge_main()
    if cmd_type == "test":
        func = getattr(args, "func", None)
        if func:
            return func(args)
        parser.parse_args(["test", "--help"])
        return 0

    try:
        return doc_forge_main()
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

def create_main_parser() -> argparse.ArgumentParser:
    """
    Creates the main parser with subcommands for documentation and testing.
    Returns a fully configured argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Doc Forge - Transform code into interconnected documentation "
            "under Eidosian principles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--version", "-V", action="store_true", help="Show version info")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command_type", help="Select command domain")

    docs_parser = subparsers.add_parser("docs", help="Documentation commands")
    docs_parser.add_subparsers(dest="command", help="Docs subcommand")
    create_doc_forge_parser()

    test_parser = subparsers.add_parser("test", help="Testing commands")
    # Type annotation and proper handling for test subcommands
    test_subparsers = test_parser.add_subparsers(dest="command", help="Test subcommand")
    add_test_subparsers(test_subparsers)

    return parser

if __name__ == "__main__":
    sys.exit(main())
