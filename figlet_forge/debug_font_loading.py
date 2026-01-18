#!/usr/bin/env python3

"""
Debug script for identifying issues with font loading in Figlet Forge.
This script provides detailed diagnostics about the font loading process.
"""

import os
import sys
from pathlib import Path

# Add the source directory to path for imports
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import the figlet components
try:
    from figlet_forge.core.figlet_font import FigletFont
    from figlet_forge.figlet import Figlet
except ImportError as e:
    print(f"Error importing Figlet modules: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


def check_font_directories():
    """Check for font directories and report findings."""
    from figlet_forge.version import SHARED_DIRECTORY

    print("\n=== FONT DIRECTORIES ===")

    search_dirs = [
        Path(SHARED_DIRECTORY) / "fonts",
        Path(__file__).parent / "src" / "figlet_forge" / "fonts",
        Path.home() / ".figlet_forge" / "fonts",
        Path("/usr/share/figlet"),
        Path("/usr/local/share/figlet"),
    ]

    for directory in search_dirs:
        exists = directory.exists()
        status = "EXISTS" if exists else "NOT FOUND"
        files = list(directory.glob("*.flf")) if exists else []
        print(f"{directory}: {status} - Contains {len(files)} .flf files")
        if files and len(files) <= 5:  # Only show if there are a few files
            for file in files:
                print(f"  - {file.name}")


def test_font_loading(font_name):
    """Test loading a specific font and report results."""
    print(f"\n=== TESTING FONT: {font_name} ===")

    try:
        font = FigletFont(font_name)
        print(f"SUCCESS: Font '{font_name}' loaded")
        print(f"Height: {font.height}")
        print(f"Num chars: {len(font.chars)}")
        return True
    except Exception as e:
        print(f"ERROR: Could not load font '{font_name}': {e}")
        return False


def test_rendering(text, font_name):
    """Test rendering text with the specified font."""
    print(f"\n=== TESTING RENDERING with font '{font_name}' ===")

    try:
        fig = Figlet(font=font_name)
        result = fig.renderText(text)
        print("RENDERED OUTPUT:")
        print("----------------")
        print(result)
        print("----------------")
        return True
    except Exception as e:
        print(f"ERROR: Could not render with font '{font_name}': {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main diagnostics script."""
    print("=== FIGLET FORGE DIAGNOSTICS ===")
    print(f"Working directory: {os.getcwd()}")

    # Check font directories
    check_font_directories()

    # Test standard font
    test_font_loading("standard")

    # Test rendering
    test_rendering("Hello", "standard")

    # Also try slant font
    print("\n=== ADDITIONAL TESTS ===")
    test_font_loading("slant")
    test_rendering("Hello", "slant")


if __name__ == "__main__":
    main()
