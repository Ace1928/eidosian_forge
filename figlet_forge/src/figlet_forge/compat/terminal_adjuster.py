"""
Compatibility module for terminal environment detection and adaptation.

This module provides utilities for detecting terminal capabilities,
adjusting output to match terminal constraints, and ensuring consistent
behavior across different terminal environments.
"""

import os
import platform
import shutil
import struct
import sys
from typing import Optional, Tuple

# Terminal capability constants
CAPS_NONE = 0
CAPS_COLOR = 1 << 0
CAPS_BOLD = 1 << 1
CAPS_ITALIC = 1 << 2
CAPS_UNDERLINE = 1 << 3
CAPS_STRIKETHROUGH = 1 << 4
CAPS_UNICODE = 1 << 5
CAPS_256_COLOR = 1 << 6
CAPS_TRUECOLOR = 1 << 7
CAPS_TRUE_COLOR = 1 << 7  # Alias for CAPS_TRUECOLOR for compatibility
CAPS_WIDE_CHARS = 1 << 8

# Terminal type constants
TERM_TYPE_UNKNOWN = 0
TERM_TYPE_DUMB = 1
TERM_TYPE_ANSI = 2
TERM_TYPE_WINDOWS = 3
TERM_TYPE_NO_COLOR = 4


class TerminalAdjuster:
    """
    Handles detection and adaptation for different terminal environments.

    This class detects terminal capabilities and provides utilities for
    adapting output to match the capabilities of the current terminal.
    """

    def __init__(self, force_capabilities: int = 0, disable_capabilities: int = 0):
        """
        Initialize the TerminalAdjuster.

        Args:
            force_capabilities: Capabilities to force enable
            disable_capabilities: Capabilities to force disable
        """
        self.force_capabilities = force_capabilities
        self.disable_capabilities = disable_capabilities
        self._capabilities = None
        self._dimensions = None
        self._term_type = self._detect_terminal_type()
        self._color_depth = self._detect_color_depth()

    def _detect_terminal_type(self) -> int:
        """
        Detect the type of terminal we're running in.

        Returns:
            Terminal type constant
        """
        # Check for CI environments - they usually support ANSI
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            return TERM_TYPE_ANSI

        # Check for NO_COLOR standard
        if os.environ.get("NO_COLOR") is not None:
            return TERM_TYPE_NO_COLOR

        # Check for dumb terminal
        if os.environ.get("TERM") == "dumb":
            return TERM_TYPE_DUMB

        # Check if stdout is a TTY
        if not sys.stdout.isatty():
            if platform.system() == "Windows":
                return TERM_TYPE_WINDOWS
            return TERM_TYPE_UNKNOWN

        # Windows-specific checks
        if platform.system() == "Windows":
            # Windows Terminal, VSCode, and modern terminals support ANSI
            if (
                os.environ.get("WT_SESSION")
                or os.environ.get("TERM_PROGRAM") in ("vscode", "alacritty")
                or hasattr(sys, "getwindowsversion")
                and sys.getwindowsversion().major >= 10
            ):
                return TERM_TYPE_ANSI
            return TERM_TYPE_WINDOWS

        # Most Unix terminals support ANSI
        return TERM_TYPE_ANSI

    @property
    def capabilities(self) -> int:
        """
        Get the capabilities of the current terminal.

        Returns:
            Bitfield of supported capabilities
        """
        if self._capabilities is not None:
            return self._capabilities

        caps = CAPS_NONE

        # Check for forced/disabled capabilities
        if self.force_capabilities:
            return self.force_capabilities

        # Check if output is being redirected
        if not sys.stdout.isatty():
            # Minimal capabilities for redirected output
            self._capabilities = CAPS_NONE
            return CAPS_NONE

        # Platform checks
        system = platform.system()

        # Check for color support
        if _detect_color_support():
            caps |= CAPS_COLOR

            # Check for extended color capabilities
            if _detect_256_color_support():
                caps |= CAPS_256_COLOR

                # Check for true color
                if _detect_true_color_support():
                    caps |= CAPS_TRUECOLOR

        # Text formatting supports
        term = os.environ.get("TERM", "").lower()
        if term and term not in ("dumb", "unknown"):
            caps |= CAPS_BOLD

            # Modern terminals usually support these
            if "xterm" in term or "rxvt" in term or "256color" in term:
                caps |= CAPS_ITALIC | CAPS_UNDERLINE

                # More advanced formatting
                if "xterm" in term and "-256color" in term:
                    caps |= CAPS_STRIKETHROUGH

        # Unicode support
        if _detect_unicode_support():
            caps |= CAPS_UNICODE

            # Check for wide character support (CJK, etc)
            if self._check_wide_char_support():
                caps |= CAPS_WIDE_CHARS

        # Apply capability disables
        caps &= ~self.disable_capabilities

        self._capabilities = caps
        return caps

    def _detect_capabilities(self) -> int:
        """
        Detect terminal capabilities based on environment.

        Returns:
            Bitfield of capabilities
        """
        caps = CAPS_NONE

        # Unknown or dumb terminals have no capabilities
        if self._term_type in (TERM_TYPE_UNKNOWN, TERM_TYPE_DUMB, TERM_TYPE_NO_COLOR):
            return caps

        # Windows terminal capabilities
        if self._term_type == TERM_TYPE_WINDOWS:
            # Basic color support in modern Windows
            if (
                hasattr(sys, "getwindowsversion")
                and sys.getwindowsversion().major >= 10
            ):
                caps |= CAPS_COLOR

                # Windows Terminal and other modern terminals have more capabilities
                if os.environ.get("WT_SESSION"):
                    caps |= (
                        CAPS_UNICODE
                        | CAPS_WIDE_CHARS
                        | CAPS_BOLD
                        | CAPS_ITALIC
                        | CAPS_TRUE_COLOR
                    )
            return caps

        # ANSI terminal capabilities
        if self._term_type == TERM_TYPE_ANSI:
            term = os.environ.get("TERM", "").lower()

            # Basic formatting
            caps |= CAPS_COLOR | CAPS_BOLD

            # Terminal with explicit 256-color support
            if "256color" in term:
                caps |= CAPS_256_COLOR

            # Check for Unicode support via locale
            if os.environ.get("LANG", "").lower().endswith(("utf-8", "utf8")):
                caps |= CAPS_UNICODE | CAPS_WIDE_CHARS

            # Advanced formatting for modern terminals
            if "xterm" in term:
                caps |= CAPS_ITALIC

            # True color support
            if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
                caps |= CAPS_TRUE_COLOR

        return caps

    def _check_wide_char_support(self) -> bool:
        """
        Check if terminal likely supports wide (CJK) characters.

        Returns:
            True if wide character support is likely
        """
        # Most terminals that support Unicode also support wide chars
        if "LANG" in os.environ and (
            "UTF" in os.environ["LANG"].upper()
            or "JP" in os.environ["LANG"].upper()
            or "KR" in os.environ["LANG"].upper()
            or "CN" in os.environ["LANG"].upper()
        ):
            return True

        # Check for CJK-friendly terminals
        term = os.environ.get("TERM", "").lower()
        if "cjk" in term or "utf" in term:
            return True

        return False

    def _detect_color_depth(self) -> int:
        """
        Detect the terminal's color depth.

        Returns:
            0, 16, 256, or 16777216 (24-bit)
        """
        # No color support
        if self._term_type in (TERM_TYPE_DUMB, TERM_TYPE_NO_COLOR) or not (
            self._capabilities & CAPS_COLOR
        ):
            return 0

        # True color support
        if "COLORTERM" in os.environ and os.environ["COLORTERM"] in (
            "truecolor",
            "24bit",
        ):
            return 16777216

        # Check for true color capability flag
        if self._capabilities & CAPS_TRUE_COLOR:
            return 16777216

        # Check for 256 color terminal
        term = os.environ.get("TERM", "")
        if "256color" in term or self._capabilities & CAPS_256_COLOR:
            return 256

        # Default to 16 colors
        return 16

    def has_capability(self, capability: int) -> bool:
        """
        Check if the terminal supports a specific capability.

        Args:
            capability: Capability to check for

        Returns:
            True if supported, False otherwise
        """
        return bool(self.capabilities & capability)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """
        Get the current terminal dimensions.

        Returns:
            Tuple of (width, height) in characters
        """
        if self._dimensions is not None:
            return self._dimensions

        # Try environment variables first
        try:
            columns = int(os.environ.get("COLUMNS", "0"))
            lines = int(os.environ.get("LINES", "0"))
            if columns > 0 and lines > 0:
                self._dimensions = (columns, lines)
                return self._dimensions
        except (ValueError, TypeError):
            pass

        # Try using shutil.get_terminal_size
        try:
            size = shutil.get_terminal_size()
            self._dimensions = (size.columns, size.lines)
            return self._dimensions
        except (AttributeError, OSError):
            pass

        # Try platform-specific approaches
        if platform.system() == "Windows":
            self._dimensions = _get_windows_terminal_size()
        else:
            self._dimensions = _get_unix_terminal_size()

        # Default fallback
        if not self._dimensions or self._dimensions == (0, 0):
            self._dimensions = (80, 24)

        return self._dimensions

    @property
    def terminal_width(self) -> int:
        """Get the terminal width."""
        return self.dimensions[0]

    @property
    def terminal_height(self) -> int:
        """Get the terminal height."""
        return self.dimensions[1]

    @property
    def supports_color(self) -> bool:
        """Check if terminal supports color."""
        return self.has_capability(CAPS_COLOR)

    @property
    def supports_unicode(self) -> bool:
        """Check if terminal supports Unicode."""
        return self.has_capability(CAPS_UNICODE)

    @property
    def supports_true_color(self) -> bool:
        """Check if terminal supports true color (24-bit)."""
        return self.has_capability(CAPS_TRUE_COLOR)

    @property
    def color_depth(self) -> int:
        """Get the terminal's color depth."""
        return self._color_depth

    def wrap_text(self, text: str, width: Optional[int] = None) -> str:
        """
        Wrap text to fit within terminal width.

        Args:
            text: Text to wrap
            width: Maximum width (default: terminal width)

        Returns:
            Wrapped text
        """
        if width is None:
            width = self.dimensions[0]

        if width <= 0:
            width = 80

        lines = []
        for line in text.splitlines():
            if len(line) <= width:
                lines.append(line)
                continue

            # Simple word wrapping
            current_line = ""
            for word in line.split():
                if not current_line:
                    current_line = word
                elif len(current_line) + len(word) + 1 <= width:
                    current_line += " " + word
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

        return "\n".join(lines)

    def strip_formatting(self, text: str) -> str:
        """
        Strip ANSI formatting if the terminal doesn't support it.

        Args:
            text: Text with potential formatting

        Returns:
            Text with formatting removed if necessary
        """
        if not self.supports_color:
            import re

            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return ansi_escape.sub("", text)
        return text

    def adjust_output_for_terminal(self, text: str, enforce_ascii: bool = False) -> str:
        """
        Adjust output to match terminal capabilities.

        Args:
            text: Text to adjust
            enforce_ascii: Whether to force ASCII output

        Returns:
            Adjusted text
        """
        # Handle Unicode/ASCII conversion
        if enforce_ascii or not self.supports_unicode:
            # Replace Unicode with ASCII approximations
            text = self._unicode_to_ascii(text)

        # Handle color/formatting
        if not self.supports_color:
            text = self.strip_formatting(text)

        return text

    def _unicode_to_ascii(self, text: str) -> str:
        """
        Convert Unicode characters to ASCII approximations.

        Args:
            text: Unicode text

        Returns:
            ASCII approximation
        """
        # Common replacements for box-drawing and other Unicode
        replacements = {
            # Box drawing
            "─": "-",
            "│": "|",
            "┌": "+",
            "┐": "+",
            "└": "+",
            "┘": "+",
            "├": "+",
            "┤": "+",
            "┬": "+",
            "┴": "+",
            "┼": "+",
            # Bullets and arrows
            "•": "*",
            "◦": "o",
            "→": "->",
            "←": "<-",
            "▶": ">",
            "◀": "<",
            "…": "...",
            # Block elements
            "█": "#",
            "▓": "%",
            "▒": ":",
            "░": ".",
            # Other common symbols
            "☐": "[ ]",
            "☑": "[x]",
            "☒": "[X]",
            "✓": "v",
            "✔": "V",
            "✗": "x",
            "✘": "X",
            "♥": "<3",
            "♦": "<>",
            # Currency
            "€": "EUR",
            "£": "GBP",
            "¥": "JPY",
            # Other
            "©": "(c)",
            "®": "(R)",
            "™": "(TM)",
        }

        for uni, ascii in replacements.items():
            text = text.replace(uni, ascii)

        # Final fallback: just drop all non-ASCII
        return text.encode("ascii", "replace").decode("ascii")

    def reset_capabilities(self) -> None:
        """Reset cached capabilities and dimensions."""
        self._capabilities = None
        self._dimensions = None


# Helper functions
def _detect_color_support() -> bool:
    """Detect if terminal supports color output."""
    # Check for NO_COLOR environment variable
    if os.environ.get("NO_COLOR") is not None:
        return False

    # Check for FORCE_COLOR
    if os.environ.get("FORCE_COLOR") is not None:
        return True

    # Platform-specific checks
    if platform.system() == "Windows":
        # Windows 10 build 14931+ supports ANSI colors
        if sys.getwindowsversion().build >= 14931:
            return True

        # Check for color-supporting terminal emulators
        if os.environ.get("TERM_PROGRAM") in ("vscode", "conemu", "cmder", "alacritty"):
            return True

        return False
    else:
        # Unix-like systems
        term = os.environ.get("TERM", "")
        color_terms = (
            "xterm",
            "xterm-color",
            "xterm-256color",
            "linux",
            "screen",
            "screen-256color",
            "vt100",
            "color",
            "ansi",
        )

        if any(t in term for t in color_terms) or os.environ.get("COLORTERM"):
            return True

        return False


def _detect_256_color_support() -> bool:
    """Detect if terminal supports 256 colors."""
    if platform.system() == "Windows":
        # Windows 10 build 14931+ supports 256 colors
        if sys.getwindowsversion().build >= 14931:
            return True
        return False

    # Check terminal type
    term = os.environ.get("TERM", "")
    if "256color" in term or "xterm" in term:
        return True

    # Check for colorterm indicating 256 color support
    colorterm = os.environ.get("COLORTERM", "")
    if "256" in colorterm:
        return True

    return False


def _detect_true_color_support() -> bool:
    """Detect if terminal supports true color (24-bit)."""
    colorterm = os.environ.get("COLORTERM", "")
    if "truecolor" in colorterm or "24bit" in colorterm:
        return True

    # Check for terminals known to support true color
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program in ("iTerm.app", "vscode", "alacritty", "wezterm"):
        return True

    return False


def _detect_unicode_support() -> bool:
    """Detect if terminal supports Unicode."""
    if platform.system() == "Windows":
        # Windows 10 supports Unicode well
        if sys.getwindowsversion().major >= 10:
            return True

    # Check locale settings
    lang = os.environ.get("LANG", "")
    if "utf" in lang.lower():
        return True

    return False


def _get_windows_terminal_size() -> Tuple[int, int]:
    """Get terminal size on Windows."""
    try:
        # Try using get_terminal_size for modern Python
        size = shutil.get_terminal_size()
        return (size.columns, size.lines)
    except (AttributeError, OSError):
        try:
            # Fallback for older Python versions
            import ctypes

            h = ctypes.windll.kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            csbi = ctypes.create_string_buffer(22)
            res = ctypes.windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)

            if res:
                (_, _, _, _, _, left, top, right, bottom, _, _) = struct.unpack(
                    "hhhhHhhhhhh", csbi.raw
                )
                width = right - left + 1
                height = bottom - top + 1
                return (width, height)
        except (ImportError, AttributeError):
            pass

    return (80, 24)  # Default fallback


def _get_unix_terminal_size() -> Tuple[int, int]:
    """Get terminal size on Unix-like systems."""
    try:
        # Try using get_terminal_size for modern Python
        size = shutil.get_terminal_size()
        return (size.columns, size.lines)
    except (AttributeError, OSError):
        try:
            # Fallback using ioctl
            import fcntl
            import termios

            with open(os.ctermid(), "rb") as fd:
                size = struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))
                return (size[1], size[0])
        except (OSError, ImportError):
            pass

    return (80, 24)  # Default fallback


# Initialize singleton
terminal = TerminalAdjuster()


# Convenience functions for external use
def get_terminal_size() -> Tuple[int, int]:
    """
    Get terminal dimensions.

    Returns:
        Tuple of (width, height) in characters
    """
    return terminal.dimensions


def supports_color() -> bool:
    """
    Check if terminal supports color.

    Returns:
        True if color is supported, False otherwise
    """
    return terminal.supports_color


def supports_unicode() -> bool:
    """
    Check if terminal supports Unicode.

    Returns:
        True if Unicode is supported, False otherwise
    """
    return terminal.supports_unicode


def adjust_output(text: str, enforce_ascii: bool = False) -> str:
    """
    Adjust output for terminal capabilities.

    Args:
        text: Text to adjust
        enforce_ascii: Whether to force ASCII

    Returns:
        Adjusted text
    """
    return terminal.adjust_output_for_terminal(text, enforce_ascii)


# Export constants and functions for backwards compatibility
__all__ = [
    "TerminalAdjuster",
    "CAPS_NONE",
    "CAPS_COLOR",
    "CAPS_BOLD",
    "CAPS_ITALIC",
    "CAPS_UNDERLINE",
    "CAPS_STRIKETHROUGH",
    "CAPS_UNICODE",
    "CAPS_256_COLOR",
    "CAPS_TRUECOLOR",
    "CAPS_TRUE_COLOR",
    "CAPS_WIDE_CHARS",
    "TERM_TYPE_UNKNOWN",
    "TERM_TYPE_DUMB",
    "TERM_TYPE_ANSI",
    "TERM_TYPE_WINDOWS",
    "TERM_TYPE_NO_COLOR",
    "get_terminal_size",
    "supports_color",
    "supports_unicode",
    "adjust_output",
]
