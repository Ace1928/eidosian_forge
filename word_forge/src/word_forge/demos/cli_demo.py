from eidosian_core import eidosian

"""Demonstration of Word Forge CLI capabilities.

This module provides interactive demonstrations of the Word Forge
command-line interface, showcasing various commands and options:

- Starting the processing pipeline
- Managing graph operations
- Running vector indexing and search
- Managing conversations
- Configuring and monitoring workers

Usage:
    python -m word_forge.demos.cli_demo

For the main CLI entry point, use:
    word_forge --help
"""


@eidosian()
def main() -> None:
    """Run CLI demonstration showcasing available commands.

    Displays an overview of Word Forge CLI capabilities and
    provides examples of common usage patterns.
    """
    print("Word Forge CLI Demonstration")
    print("=" * 40)
    print()
    print("Word Forge provides a powerful CLI for lexical processing.")
    print()
    print("Main Commands:")
    print("  word_forge start [WORDS...]      Start processing seed words")
    print("  word_forge graph build           Build the semantic graph")
    print("  word_forge graph visualize       Generate visualization")
    print("  word_forge vector index          Run vector indexing")
    print("  word_forge vector search QUERY   Search for similar terms")
    print("  word_forge conversation start    Start a new conversation")
    print("  word_forge conversation list     List conversations")
    print("  word_forge emotion annotate      Run emotion annotation")
    print("  word_forge demo full             Run full demo pipeline")
    print("  word_forge setup-nltk            Download NLTK data")
    print()
    print("Common Options:")
    print("  --version, -V    Show version")
    print("  --help, -h       Show help")
    print("  --quiet, -q      Suppress output")
    print("  --verbose, -v    Enable debug output")
    print("  --config FILE    Load configuration file")
    print()
    print("For detailed help on any command, use:")
    print("  word_forge COMMAND --help")


if __name__ == "__main__":
    main()
