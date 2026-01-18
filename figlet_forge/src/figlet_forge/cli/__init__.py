"""
Command-line interface for Figlet Forge.

Provides tools for interacting with the Figlet Forge library
through the command line.
"""

import sys
from typing import List, Optional, Union

# Import the sample module to ensure it's available
from . import sample

# Import the main CLI function from main.py
from .main import main
from .showcase import generate_showcase

__all__ = ["main", "sample", "generate_showcase"]

# Provide direct execution capability
if __name__ == "__main__":
    sys.exit(main())
