#!/usr/bin/env python

"""
╔═══════════════════════════════════════════════════════════════════════════╗
║   ███████╗██╗ ██████╗ ██╗     ███████╗████████╗    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗   ║
║   ██╔════╝██║██╔════╝ ██║     ██╔════╝╚══██╔══╝    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝   ║
║   █████╗  ██║██║  ███╗██║     █████╗     ██║       █████╗  ██║   ██║██████╔╝██║  ███╗█████╗     ║
║   ██╔══╝  ██║██║   ██║██║     ██╔══╝     ██║       ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝     ║
║   ██║     ██║╚██████╔╝███████╗███████╗   ██║       ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗   ║
║   ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝       ╚═╝       ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ║
║                                                                                                   ║
║ ╭────────────────────────────────────────────────────────────────────────────────────────────╮   ║
║ │           TEXT CRYSTALLIZATION ENGINE - DIGITAL TYPOGRAPHY FORGE                           │   ║
║ ╰────────────────────────────────────────────────────────────────────────────────────────────╯   ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝

Figlet Forge: An Eidosian reimplementation extending pyfiglet with colorized ANSI support,
Unicode rendering and intelligent fallbacks while maintaining backward compatibility.
"""

import importlib.util
import logging
import sys
from typing import Any, Dict, List, Optional, Union

# Import color support
from .color import ColorMode, ColorScheme, colored_format, parse_color

# Import core functionality
from .core.exceptions import (
    CharNotPrinted,
    FigletError,
    FontError,
    FontNotFound,
    InvalidColor,
)
from .core.figlet_font import FigletFont
from .figlet import Figlet, print_figlet
from .figlet_string import FigletString

# Import render functionality
from .render.figlet_engine import RenderEngine

# Import version information first to avoid circular imports
from .version import (
    COLOR_CODES,
    DEFAULT_FONT,
    RESET_COLORS,
    __author__,
    __author_email__,
    __description__,
    __docs_url__,
    __github_url__,
    __package_name__,
    __version__,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Package exports
__all__ = [
    # Main classes
    "Figlet",
    "FigletString",
    "FigletFont",
    # Convenience functions
    "print_figlet",
    "colored_format",
    # Color support
    "ColorMode",
    "ColorScheme",
    "parse_color",
    # Rendering
    "RenderEngine",
    # Constants
    "DEFAULT_FONT",
    "COLOR_CODES",
    "RESET_COLORS",
    # Exceptions
    "FigletError",
    "FontNotFound",
    "FontError",
    "CharNotPrinted",
    "InvalidColor",
    # Version info
    "__version__",
    "__author__",
    "__description__",
]
