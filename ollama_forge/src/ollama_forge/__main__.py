#!/usr/bin/env python3
# 🌀 Entry point infused with Eidosian flow – precise and impactful! ⚡
"""
Ollama Forge Module Execution Point

This module serves as the direct execution entry point when the package is run with:
`python -m ollama_forge`

Following Eidosian principles of:
- Contextual Integrity: Every component has a precise purpose
- Flow Like a River: Seamless execution path
- Structure as Control: Clear organization and responsibility delegation
"""

import argparse
import sys
from typing import Callable, Dict, List, Optional

# Import CLI components
from ollama_forge.cli import main as cli_main
from ollama_forge.cli.commands import (
    chat_command,
    embedding_command,
    generate_command,
    health_command,
    models_command,
)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command arguments with Eidosian precision and clarity.
    Each argument serves a purpose; no waste, no dilution.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Parsed arguments with perfect structure
    """
    parser = argparse.ArgumentParser(
        description="🔥 Ollama Forge: Precision-crafted LLM toolkit",
        epilog="📚 For detailed documentation: https://ollama-forge.readthedocs.io",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Command structure - elegant subparser architecture
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Generate command - text synthesis with intelligence
    generate_parser = subparsers.add_parser(
        "generate", help="Generate text from a prompt"
    )
    generate_parser.add_argument("prompt", help="Prompt to generate from")
    generate_parser.add_argument(
        "--model",
        "-m",
        default="deepseek-r1:1.5b",
        help="Model to use (default: %(default)s)",
    )
    generate_parser.add_argument(
        "--stream", "-s", action="store_true", help="Stream output token by token"
    )

    # Chat command - flowing dialogue with AI
    chat_parser = subparsers.add_parser("chat", help="Chat with the model")
    chat_parser.add_argument("message", nargs="?", help="Initial message to send")
    chat_parser.add_argument(
        "--model",
        "-m",
        default="qwen2.5:3b-Instruct",
        help="Model to use (default: %(default)s)",
    )
    chat_parser.add_argument(
        "--session", action="store_true", help="Start an interactive session"
    )

    # Models command - knowledge architecture
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument(
        "--refresh", action="store_true", help="Refresh model list from server"
    )

    # Embedding command - vector transformation
    embed_parser = subparsers.add_parser("embed", help="Create embeddings from text")
    embed_parser.add_argument("text", help="Text to embed")
    embed_parser.add_argument(
        "--model",
        "-m",
        default="deepseek-r1:0.5b",
        help="Model to use (default: %(default)s)",
    )

    # Health check - system self-awareness
    subparsers.add_parser("health", help="Check Ollama server health")

    return parser.parse_args(args)


def command_router() -> Dict[str, Callable]:
    """
    Maps commands to their handler functions.
    Follows the 'Structure as Control' principle with a single source of routing.

    Returns:
        Dictionary mapping command names to handler functions
    """
    return {
        "generate": generate_command,
        "chat": chat_command,
        "embed": embedding_command,
        "models": models_command,
        "health": health_command,
    }


def main() -> int:
    """
    Initiates command flow with Eidosian elegance.
    Each operation precisely executed, each error handled with wit.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        args = parse_args()

        if not args.command:
            print("⚡ Ollama Forge awaits your command!")
            print("Try 'python -m ollama_forge generate \"Hello, world!\"'")
            return 0

        # Route command through elegant dispatch system
        router = command_router()
        if args.command in router:
            return router[args.command](args)
        else:
            print(
                f"🤔 Command '{args.command}' exists in a parallel universe, not this one."
            )
            return 1

    except KeyboardInterrupt:
        print("\n🛑 Flow interrupted - exiting with grace and poise.")
        return 130
    except Exception as e:
        print(f"❌ Error encountered: {str(e)}")
        print("Even the most elegant systems have moments of reflection! 🪞")
        return 1


def module_entry(args: Optional[List[str]] = None) -> int:
    """
    Pure entry point function for programmatic control.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success)
    """
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
