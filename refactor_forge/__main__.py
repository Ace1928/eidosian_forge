"""
Main Entry Point - Direct Package Execution 🚀

Enables running the package directly using `python -m eidosian_refactor`,
providing a convenient entry point for command-line usage.
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
