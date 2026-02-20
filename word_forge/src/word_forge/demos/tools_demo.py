from eidosian_core import eidosian

"""Demonstration of Word Forge utility tools.

This module showcases the various utility tools and helper functions
available in Word Forge for common operations:

- NLTK data management (downloading corpora)
- Path and directory utilities
- Configuration validation helpers
- Logging setup utilities

Usage:
    python -m word_forge.demos.tools_demo
"""


@eidosian()
def main() -> None:
    """Run utility tools demonstration.

    Demonstrates the usage of various Word Forge utility functions
    including NLTK setup, path management, and validation tools.
    """
    print("Word Forge Tools Demonstration")
    print("=" * 40)
    print()

    # Demonstrate NLTK utilities
    print("1. NLTK Utilities")
    print("-" * 40)
    print("Word Forge includes NLTK data management utilities.")
    print("To download required NLTK data, use:")
    print("  word_forge setup-nltk")
    print("Or programmatically:")
    print("  from word_forge.utils.nltk_utils import ensure_nltk_data")
    print("  downloaded = ensure_nltk_data()")
    print()

    # Demonstrate configuration utilities
    print("2. Configuration Utilities")
    print("-" * 40)
    print("Configuration can be accessed and validated:")
    print("  from word_forge.config import config")
    print("  errors = config.validate_all()")
    print("  config.export_to_file('config.json')")
    print()

    # Demonstrate path utilities
    print("3. Path Constants")
    print("-" * 40)
    print("Common paths are available as constants:")
    print("  from word_forge.configs.config_essentials import DATA_ROOT")
    print("  print(DATA_ROOT)  # Default data directory")
    print()

    print("For more details, explore the word_forge.utils module.")


if __name__ == "__main__":
    main()
