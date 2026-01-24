"""
Figlet Forge version information and constants.

This module provides version information and shared constants for the
Figlet Forge package.
"""

import os
import sys
from pathlib import Path

# Version information
__version__ = "0.1.0"
VERSION = __version__
__package_name__ = "figlet-forge"
PROGRAM_NAME = "figlet-forge"
__author__ = "Lloyd Handyside"
__author_email__ = "ace1928@gmail.com"
__description__ = "Advanced FIGlet text rendering with color and Unicode support"
__github_url__ = "https://github.com/Ace1928/figlet_forge"
__docs_url__ = "https://github.com/Ace1928/figlet_forge/docs"

# Font configuration
DEFAULT_FONT = "standard"

# Color configurations
RESET_COLORS = "\033[0m"  # Color reset code

# Width defaults
DEFAULT_WIDTH = 80

# Figlet-specific constants
SHARED_DIRECTORY = "/usr/share"  # Base directory for shared files

# Color codes for ANSI terminal colors
COLOR_CODES = {
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "LIGHT_BLACK": "\033[90m",
    "LIGHT_RED": "\033[91m",
    "LIGHT_GREEN": "\033[92m",
    "LIGHT_YELLOW": "\033[93m",
    "LIGHT_BLUE": "\033[94m",
    "LIGHT_MAGENTA": "\033[95m",
    "LIGHT_CYAN": "\033[96m",
    "LIGHT_WHITE": "\033[97m",
}

# Alias for backward compatibility
COLORS = COLOR_CODES

# Background color codes
BG_COLOR_CODES = {
    "BLACK": "\033[40m",
    "RED": "\033[41m",
    "GREEN": "\033[42m",
    "YELLOW": "\033[43m",
    "BLUE": "\033[44m",
    "MAGENTA": "\033[45m",
    "CYAN": "\033[46m",
    "WHITE": "\033[47m",
    "LIGHT_BLACK": "\033[100m",
    "LIGHT_RED": "\033[101m",
    "LIGHT_GREEN": "\033[102m",
    "LIGHT_YELLOW": "\033[103m",
    "LIGHT_BLUE": "\033[104m",
    "LIGHT_MAGENTA": "\033[105m",
    "LIGHT_CYAN": "\033[106m",
    "LIGHT_WHITE": "\033[107m",
}

# Directory structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)

# Detect installation environment
try:
    import site

    USER_SITE = site.getusersitepackages()
    SHARED_DIRECTORY = os.path.dirname(os.path.dirname(USER_SITE))
except (ImportError, AttributeError):
    # Fallback for when site module is not available
    USER_SITE = os.path.expanduser("~/.local/lib/python")
    SHARED_DIRECTORY = "/usr/share"

# Font-related settings
FONT_EXTENSIONS = [".flf", ".tlf"]
FONT_SEARCH_PATHS = [
    Path(__file__).parent / "fonts",  # Package fonts
    Path(SHARED_DIRECTORY) / "figlet" / "fonts",  # System fonts (Unix-like)
    Path(os.path.expanduser("~")) / ".figlet_forge" / "fonts",  # User fonts
]

# Add OS-specific paths
if sys.platform == "win32":
    FONT_SEARCH_PATHS.extend(
        [
            Path(os.environ.get("APPDATA", "")) / "figlet_forge" / "fonts",
            Path(os.environ.get("PROGRAMFILES", "")) / "Figlet" / "fonts",
        ]
    )
elif sys.platform == "darwin":
    FONT_SEARCH_PATHS.extend(
        [
            Path("/opt/local/share/figlet/fonts"),
            Path.home() / "Library" / "figlet" / "fonts",
        ]
    )
