"""
Compatibility layer for various terminal environments and external libraries.

This package ensures that Figlet Forge works consistently across different
environments and provides backward compatibility with other figlet libraries.
"""

# Terminal capability detection
# Color adjustment
from ..figlet import Figlet
from ..core.exceptions import (
    CharNotPrinted,
    FigletError,
    FontError,
    FontNotFound,
    InvalidColor,
)
from ..version import __version__
from .colour_adjuster import (
    ColourAdjuster,
    adapt_colors_to_terminal,
    strip_colors,
)

# Encoding handling
from .encoding_adjuster import (
    EncodingAdjuster,
    decode_bytes,
    encode_text,
    get_system_encoding,
    supports_utf8,
)
from .figlet_compat import figlet_format, renderText
from .terminal_adjuster import (
    TerminalAdjuster,
    adjust_output,
    get_terminal_size,
    supports_color,
    supports_unicode,
)

# Default settings for compatibility
DEFAULT_FONT = "standard"
DEFAULT_DIRECTION = "auto"
DEFAULT_JUSTIFY = "auto"
DEFAULT_WIDTH = 80
VERSION = __version__

__all__ = [
    # Figlet compatibility
    "Figlet",
    "figlet_format",
    "renderText",
    "DEFAULT_FONT",
    "DEFAULT_DIRECTION",
    "DEFAULT_JUSTIFY",
    "DEFAULT_WIDTH",
    "VERSION",
    # Exceptions for backward compatibility
    "FigletError",
    "FontNotFound",
    "FontError",
    "CharNotPrinted",
    "InvalidColor",
    # Terminal adjusters
    "TerminalAdjuster",
    "get_terminal_size",
    "supports_color",
    "supports_unicode",
    "adjust_output",
    # Color adjusters
    "ColourAdjuster",
    "adapt_colors_to_terminal",
    "strip_colors",
    # Encoding adjusters
    "EncodingAdjuster",
    "get_system_encoding",
    "supports_utf8",
    "decode_bytes",
    "encode_text",
]
