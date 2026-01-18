"""
Command Line Interface for Ollama Forge. 🎯🌟
Eidosian synergy: minimal redundancy, immediate utility.
"""

from typing import List, Optional

from ollama_forge.src.ollama_forge.cli.commands import (
    create_parser,
    handle_chat,
    handle_chat_session,
    handle_embed,
    handle_generate,
    handle_list,
    handle_pull,
)
from ollama_forge.src.ollama_forge.core import OllamaClient


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI with perfect flow control.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        # Ensure correct usage of base_url
        client = OllamaClient(base_url=parsed_args.api_url)
        handlers = {
            "generate": handle_generate,
            "chat": handle_chat,
            "chat-session": handle_chat_session,
            "embed": handle_embed,
            "list": handle_list,
            "pull": handle_pull,
        }
        handler = handlers.get(parsed_args.command)
        if handler:
            return handler(parsed_args, client)
        else:
            print(f"Unknown command: {parsed_args.command}")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
