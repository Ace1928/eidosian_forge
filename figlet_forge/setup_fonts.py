#!/usr/bin/env python3

"""
Setup script to install standard fonts for Figlet Forge.

This script ensures that the essential fonts are available
for the package to function correctly.
"""

import os
import shutil
import sys
from pathlib import Path

# Define essential fonts - minimum set required for compatibility
ESSENTIAL_FONTS = ["standard", "slant", "small", "big", "mini"]


def main():
    """Set up the fonts directory with essential fonts."""
    package_root = Path(__file__).parent
    fonts_dir = package_root / "src" / "figlet_forge" / "fonts"

    # Create fonts directory if needed
    fonts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up fonts directory: {fonts_dir}")

    # Check system font directories
    system_font_dirs = [
        Path("/usr/share/figlet"),
        Path("/usr/local/share/figlet"),
    ]

    # Add any OS-specific directories
    if sys.platform == "win32":
        # Windows font locations
        system_font_dirs.extend(
            [
                Path(os.environ.get("APPDATA", "")) / "figlet" / "fonts",
                Path(os.environ.get("PROGRAMFILES", "")) / "Figlet" / "fonts",
            ]
        )
    elif sys.platform == "darwin":
        # macOS font locations
        system_font_dirs.extend(
            [
                Path("/opt/local/share/figlet"),
                Path.home() / "Library" / "figlet" / "fonts",
            ]
        )
    else:
        # Additional Linux/Unix font locations
        system_font_dirs.extend(
            [
                Path("/opt/figlet/fonts"),
                Path.home() / ".figlet" / "fonts",
            ]
        )

    fonts_found = False
    fonts_missing = []

    # Copy essential fonts from system directories
    print("\n✨ Checking for essential fonts...\n")
    for font_name in ESSENTIAL_FONTS:
        font_filename = f"{font_name}.flf"
        dest_file = fonts_dir / font_filename

        # Check if font already exists in destination
        if dest_file.exists():
            print(f"✓ Font already installed: {font_name}")
            fonts_found = True
            continue

        # Try to find font in system directories
        font_found = False
        for sys_dir in system_font_dirs:
            src_file = sys_dir / font_filename
            if src_file.exists():
                try:
                    shutil.copy2(src_file, dest_file)
                    print(f"✓ Installed font: {font_name} from {sys_dir}")
                    font_found = True
                    fonts_found = True
                    break
                except Exception as e:
                    print(f"! Error copying {font_name}: {e}")

        if not font_found:
            fonts_missing.append(font_name)

    # Summary
    print("\n--- Font Setup Summary ---")
    if fonts_found:
        print("✓ Successfully set up one or more fonts.")
    else:
        print("⚠ No fonts were found or installed.")

    if fonts_missing:
        print(f"\n⚠ Missing essential fonts: {', '.join(fonts_missing)}")
        print("  You may need to install figlet fonts on your system.")
        print("  Figlet Forge will use fallback rendering for missing fonts.\n")

        # Suggestion for getting fonts
        print(
            "Suggestion: Run the get_figlet_fonts.py script to download standard fonts:"
        )
        print("  python get_figlet_fonts.py\n")

    return 0 if fonts_found else 1


if __name__ == "__main__":
    sys.exit(main())
