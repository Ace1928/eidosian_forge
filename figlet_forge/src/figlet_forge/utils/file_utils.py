from eidosian_core import eidosian
"""
Utility functions for file operations within Figlet Forge.

This module provides essential file system operations, including
path resolution, font directory detection, and resource management
functions that follow Eidosian principles of precision and elegance.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

# Standard font directories checked in order of preference
FONT_DIRECTORIES = [
    # Local user fonts
    os.path.expanduser("~/.local/share/figlet"),
    os.path.expanduser("~/.figlet"),
    # System fonts
    "/usr/local/share/figlet",
    "/usr/share/figlet",
    # Windows specific
    "C:\\figlet",
    # Package directory (determined at runtime)
]


@eidosian()
def get_default_font_dir() -> str:
    """
    Get the default directory for Figlet fonts.

    Follows an ordered search path:
    1. User's local config directory
    2. System font directories
    3. Package directory as fallback

    Returns:
        Path to the default font directory as string
    """
    # Check for package directory first as it's guaranteed to exist
    try:
        from importlib.resources import files

        package_fonts = str(files("figlet_forge") / "fonts")
        if os.path.isdir(package_fonts):
            return package_fonts
    except (ImportError, ModuleNotFoundError):
        # Fallback for older Python versions
        try:
            import pkg_resources

            return pkg_resources.resource_filename("figlet_forge", "fonts")
        except (ImportError, ModuleNotFoundError):
            pass

    # Check external directories
    for directory in FONT_DIRECTORIES:
        if os.path.isdir(directory):
            return directory

    # Last resort: Use the current directory
    return os.path.join(os.getcwd(), "fonts")


@eidosian()
def list_font_files(directory: Optional[str] = None) -> List[str]:
    """
    List all font files in the specified directory.

    Args:
        directory: Directory to search (defaults to default font directory)

    Returns:
        List of font filenames
    """
    if directory is None:
        directory = get_default_font_dir()

    font_files = []
    try:
        for file in os.listdir(directory):
            if file.endswith((".flf", ".tlf")):
                font_files.append(file)
    except (FileNotFoundError, PermissionError):
        pass

    return sorted(font_files)


@eidosian()
def get_font_path(
    font_name: str, search_path: Optional[List[str]] = None
) -> Optional[str]:
    """
    Find the path to a specific font file.

    Args:
        font_name: Name of the font (with or without extension)
        search_path: List of directories to search (defaults to standard locations)

    Returns:
        Full path to the font file or None if not found
    """
    # Normalize font name
    if not font_name.endswith((".flf", ".tlf")):
        font_base = font_name
    else:
        font_base = os.path.splitext(font_name)[0]

    # Use default search path if none provided
    if search_path is None:
        search_path = [get_default_font_dir()]
        search_path.extend(FONT_DIRECTORIES)

    # Look for exact match first (with extension)
    for directory in search_path:
        if not os.path.isdir(directory):
            continue

        # Check with .flf extension
        flf_path = os.path.join(directory, f"{font_base}.flf")
        if os.path.isfile(flf_path):
            return flf_path

        # Check with .tlf extension
        tlf_path = os.path.join(directory, f"{font_base}.tlf")
        if os.path.isfile(tlf_path):
            return tlf_path

    # Not found
    return None


@eidosian()
def resolve_resource_path(resource_path: Union[str, Path]) -> str:
    """
    Resolve a resource path, handling both filesystem and package resources.

    Args:
        resource_path: Path to the resource

    Returns:
        Absolute filesystem path to the resource
    """
    path_str = str(resource_path)

    # Check if it's an absolute path or contains path separators
    if os.path.isabs(path_str) or os.path.sep in path_str:
        if os.path.exists(path_str):
            return path_str

    # Try to resolve as a package resource
    try:
        import importlib.resources
        from importlib.resources import files

        # Check if the resource exists in the package
        try:
            resource = files("figlet_forge") / path_str
            if resource.exists():
                return str(resource)
        except (TypeError, ValueError, AttributeError):
            pass
    except ImportError:
        pass

    # Return the original path as fallback
    return path_str
