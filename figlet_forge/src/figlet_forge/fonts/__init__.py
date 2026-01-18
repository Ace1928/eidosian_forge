"""
Figlet Forge Font Resources

This package contains the default fonts for Figlet Forge, ensuring that
the package can render text even without external font files.

Font files are sourced from the original FIGlet distribution and maintain
their original licensing, which is embedded in each font file.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Set

# Configure logger for the font module
logger = logging.getLogger(__name__)

# Get the directory containing font files
FONT_DIRECTORY = Path(__file__).parent
# Define subdirectories for font categories
STANDARD_DIRECTORY = FONT_DIRECTORY / "standard"
CONTRIB_DIRECTORY = FONT_DIRECTORY / "contrib"

# Core fonts that should always be available
CORE_FONTS = ["standard", "slant", "small", "mini", "big"]

# Check if we're running from a zipped package
if not FONT_DIRECTORY.exists():
    # Alternative lookup for packaged environment
    import importlib.resources as pkg_resources

    try:
        # This works for Python 3.9+
        FONT_DIRECTORY = pkg_resources.files("figlet_forge.fonts")
        STANDARD_DIRECTORY = pkg_resources.files("figlet_forge.fonts.standard")
        CONTRIB_DIRECTORY = pkg_resources.files("figlet_forge.fonts.contrib")
    except (AttributeError, ImportError):
        # Fallback for older Python versions
        FONT_DIRECTORY = Path(pkg_resources.resource_filename("figlet_forge", "fonts"))
        STANDARD_DIRECTORY = Path(
            pkg_resources.resource_filename("figlet_forge.fonts", "standard")
        )
        CONTRIB_DIRECTORY = Path(
            pkg_resources.resource_filename("figlet_forge.fonts", "contrib")
        )


def get_font_path(font_name: str) -> Path:
    """
    Get the path to a specific font file.

    Args:
        font_name: Name of the font (without .flf extension)

    Returns:
        Path object pointing to the font file
    """
    # Search in priority order:
    # 1. Base fonts directory
    # 2. Standard fonts directory
    # 3. Contrib fonts directory

    # Normalize font name to lowercase for case-insensitive matching
    font_name_lower = font_name.lower()

    # Try to find the font file in various locations, checking both .flf and .tlf extensions
    for ext in [".flf", ".tlf"]:
        # Check main font directory first
        font_path = FONT_DIRECTORY / f"{font_name}{ext}"
        if font_path.exists():
            return font_path

        # Also check with lowercase name
        font_path = FONT_DIRECTORY / f"{font_name_lower}{ext}"
        if font_path.exists():
            return font_path

        # Next check standard directory
        font_path = STANDARD_DIRECTORY / f"{font_name}{ext}"
        if font_path.exists():
            return font_path

        # Also check with lowercase name
        font_path = STANDARD_DIRECTORY / f"{font_name_lower}{ext}"
        if font_path.exists():
            return font_path

        # Finally check contrib directory
        font_path = CONTRIB_DIRECTORY / f"{font_name}{ext}"
        if font_path.exists():
            return font_path

        # Also check with lowercase name
        font_path = CONTRIB_DIRECTORY / f"{font_name_lower}{ext}"
        if font_path.exists():
            return font_path

    # Default to the .flf extension in the main directory if not found elsewhere
    return FONT_DIRECTORY / f"{font_name}.flf"


def list_fonts() -> List[str]:
    """
    List all available fonts in the package.

    Returns:
        List of font names (without extensions)
    """
    fonts: Set[str] = set()

    try:
        # Search for fonts in each directory with priority handling
        for directory in [FONT_DIRECTORY, STANDARD_DIRECTORY, CONTRIB_DIRECTORY]:
            try:
                if directory.exists():
                    for ext in [".flf", ".tlf"]:
                        fonts.update(f.stem for f in directory.glob(f"*{ext}"))
            except Exception as e:
                logger.debug(f"Error reading from {directory}: {e}")

        # Always include core fonts in the list
        fonts.update(CORE_FONTS)
        return sorted(list(fonts))

    except Exception:
        # Fallback for packaged environments
        import importlib.resources as pkg_resources

        try:
            # This works for Python 3.9+
            package_paths = [
                "figlet_forge.fonts",
                "figlet_forge.fonts.standard",
                "figlet_forge.fonts.contrib",
            ]

            for package_path in package_paths:
                try:
                    fonts.update(
                        f.stem for f in pkg_resources.files(package_path).glob("*.flf")
                    )
                    fonts.update(
                        f.stem for f in pkg_resources.files(package_path).glob("*.tlf")
                    )
                except (ImportError, FileNotFoundError) as e:
                    logger.debug(f"Error reading from {package_path}: {e}")

            # Always include core fonts
            fonts.update(CORE_FONTS)
            return sorted(list(fonts))

        except (AttributeError, ImportError):
            # Even more basic fallback - ensure we always have the minimal set
            return sorted(CORE_FONTS)
