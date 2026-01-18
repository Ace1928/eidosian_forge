#!/usr/bin/env python3
# 🌀 Eidosian Main Entry Point
"""
Main Entry Point - The Gateway to Doc Forge

This module serves as the primary entry point for Doc Forge,
redirecting to the appropriate command handler based on user input.
Following Eidosian principles of flow and precision.
"""

import sys

# Import runner to avoid circular imports
from .run import main as run_main
from .version import get_version_string

def main() -> int:
    """
    Main entry point for Doc Forge.
    
    Returns:
        Exit code (0 for success)
    """
    # Simply delegate to run_main which handles all commands
    return run_main()

if __name__ == "__main__":
    # Display version for direct script execution
    print(f"Doc Forge v{get_version_string()}")
    print("Use 'python -m doc_forge' or 'doc-forge' to run commands.")
    sys.exit(main())
