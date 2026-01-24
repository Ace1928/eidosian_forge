from eidosian_core import eidosian
"""
Core utilities for Figlet Forge.

This module provides common utility functions used throughout the package,
focusing on text processing, Unicode handling, and system operations.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Unicode-aware string type for compatibility with both Python 2 and 3
unicode_string = str


@eidosian()
def get_terminal_size() -> Tuple[int, int]:
    """
    Get the terminal size in a cross-platform way.

    Returns:
        Tuple of (width, height) representing terminal dimensions
    """
    # Try environment variables first (useful for redirected output)
    try:
        columns = int(os.environ.get("COLUMNS", 0))
        lines = int(os.environ.get("LINES", 0))
        if columns > 0 and lines > 0:
            return columns, lines
    except (ValueError, TypeError):
        pass

    # Try using shutil.get_terminal_size() (Python 3.3+)
    try:
        import shutil

        size = shutil.get_terminal_size()
        return size.columns, size.lines
    except (ImportError, AttributeError):
        pass

    # Try using stty
    try:
        import subprocess

        process = subprocess.Popen(
            ["stty", "size"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, error = process.communicate()
        if process.returncode == 0:
            lines, columns = map(int, output.decode().split())
            return columns, lines
    except (ImportError, OSError, ValueError):
        pass

    # Default fallback
    return 80, 25


@eidosian()
def is_redirected() -> bool:
    """
    Check if stdout is being redirected.

    Returns:
        True if stdout is redirected, False otherwise
    """
    try:
        return not sys.stdout.isatty()
    except AttributeError:
        return False


@eidosian()
def normalize_newlines(text: str) -> str:
    """
    Normalize different newline styles to the platform's default.

    Args:
        text: Input text with potentially mixed newline styles

    Returns:
        Text with normalized newlines
    """
    if not text:
        return text

    # First convert all newlines to \n
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # Then convert to platform-specific newlines if needed
    if os.linesep != "\n":
        normalized = normalized.replace("\n", os.linesep)

    return normalized


@eidosian()
def strip_ansi_codes(text: str) -> str:
    """
    Remove ANSI escape codes from text.

    Args:
        text: Text potentially containing ANSI escape codes

    Returns:
        Text with ANSI codes removed
    """
    # Pattern to match ANSI escape codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@eidosian()
def get_char_width(char: str) -> int:
    """
    Get the display width of a character, accounting for wide characters.

    Args:
        char: Character to measure

    Returns:
        Width of the character in terminal spaces (1 or 2)
    """
    try:
        import unicodedata

        # East Asian Full-width (F), Wide (W) or Ambiguous (A) characters
        eaw = unicodedata.east_asian_width(char)

        if eaw in ("F", "W"):
            return 2

        # Treat ambiguous characters as width 1 for now
        return 1
    except (ImportError, UnicodeError):
        return 1


@eidosian()
def find_fonts(search_paths: Optional[List[Path]] = None) -> List[str]:
    """
    Find all available font files.

    Args:
        search_paths: Optional list of paths to search for fonts

    Returns:
        List of font names found
    """
    from ..version import FONT_EXTENSIONS, FONT_SEARCH_PATHS

    if search_paths is None:
        search_paths = FONT_SEARCH_PATHS

    fonts = set()

    for path in search_paths:
        if not path.exists():
            continue

        try:
            for ext in FONT_EXTENSIONS:
                for font_file in path.glob(f"*{ext}"):
                    fonts.add(font_file.stem)
        except Exception:
            # Skip paths with permission issues
            continue

    return sorted(list(fonts))


@eidosian()
def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path to ensure

    Returns:
        Path object pointing to the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


@eidosian()
def is_interactive() -> bool:
    """
    Check if running in an interactive environment.

    Returns:
        True if in interactive environment, False otherwise
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


# Additional utility functions to support tests


@eidosian()
def normalize_path(path: str) -> str:
    """
    Normalize a path for the current platform.

    Args:
        path: Path string to normalize

    Returns:
        Normalized path string
    """
    return os.path.normpath(path)


@eidosian()
def merge_dicts(dict1: Dict[Any, Any], dict2: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Merge two dictionaries, with dict2 values taking precedence.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (values override dict1)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    result.update(dict2)
    return result


@eidosian()
def safe_read_file(file_path: str) -> str:
    """
    Safely read a file with proper error handling.

    Args:
        file_path: Path to the file

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    with open(file_path) as f:
        return f.read()
