"""
Color support detection and adjustment for Figlet Forge.

This module provides utilities for detecting terminal color capabilities
and adjusting color output based on terminal capabilities.
"""

import os
import platform
import re
import sys
from typing import Dict


class ColorDetection:
    """
    Detect color support capabilities in the terminal environment.

    This class follows Eidosian principles of precise contextual
    awareness and adaptive functionality.
    """

    # Color depth constants
    NO_COLOR = 0
    ANSI_16_COLOR = 16
    ANSI_256_COLOR = 256
    TRUECOLOR = 16777216

    # Environment variables that affect color support
    NO_COLOR_ENV = "NO_COLOR"
    COLORTERM_ENV = "COLORTERM"
    TERM_ENV = "TERM"
    FORCE_COLOR_ENV = "FORCE_COLOR"

    def __init__(self):
        """Initialize color detection capabilities."""
        self._capabilities = self._detect_capabilities()
        self._terminal_type = self._detect_terminal_type()

    def _detect_capabilities(self) -> Dict[str, bool]:
        """
        Detect terminal capabilities related to color support.

        Returns:
            Dictionary of terminal capabilities
        """
        capabilities = {
            "color_support": True,
            "truecolor_support": False,
            "256color_support": False,
            "unicode_support": False,
            "no_color": False,
        }

        # Check for NO_COLOR environment variable
        if os.environ.get(self.NO_COLOR_ENV) is not None:
            capabilities["color_support"] = False
            capabilities["no_color"] = True
            return capabilities

        # Force color if specifically requested
        if os.environ.get(self.FORCE_COLOR_ENV) is not None:
            return {
                "color_support": True,
                "truecolor_support": True,
                "256color_support": True,
                "unicode_support": True,
                "no_color": False,
            }

        # Check if output is redirected
        if not self._is_terminal():
            capabilities["color_support"] = False
            return capabilities

        # Check for specific terminal types
        term = os.environ.get(self.TERM_ENV, "").lower()
        colorterm = os.environ.get(self.COLORTERM_ENV, "").lower()

        # Check for Unicode support
        lang = os.environ.get("LANG", "")
        capabilities["unicode_support"] = "utf" in lang.lower()

        # Check specific terminal capabilities
        if "truecolor" in colorterm or "24bit" in colorterm:
            capabilities["truecolor_support"] = True
            capabilities["256color_support"] = True
        elif "256" in colorterm or "256color" in term:
            capabilities["256color_support"] = True
        elif "dumb" in term:
            capabilities["color_support"] = False
        elif not term and platform.system() == "Windows":
            # Windows specific detection
            try:
                win_ver = sys.getwindowsversion().major
                if win_ver >= 10:
                    capabilities["color_support"] = True
                    # Windows 10 1909+ supports true color in modern terminals
                    if "WT_SESSION" in os.environ or "TERM_PROGRAM" in os.environ:
                        capabilities["truecolor_support"] = True
                        capabilities["256color_support"] = True
            except (AttributeError, OSError):
                capabilities["color_support"] = self._detect_windows_legacy_support()

        return capabilities

    def _detect_terminal_type(self) -> str:
        """
        Detect the terminal type (Windows, Linux, etc.).

        Returns:
            String identifying the terminal type
        """
        system = platform.system()

        # Check for CI environments first
        if os.environ.get("CI") is not None:
            if os.environ.get("GITHUB_ACTIONS") is not None:
                return "GitHub Actions"
            elif os.environ.get("TRAVIS") is not None:
                return "Travis CI"
            elif os.environ.get("JENKINS_URL") is not None:
                return "Jenkins"
            return "CI Environment"

        # OS-specific terminal types
        if system == "Windows":
            if os.environ.get("WT_SESSION") is not None:
                return "Windows Terminal"
            elif os.environ.get("TERM_PROGRAM") == "vscode":
                return "VS Code Terminal"
            return "Windows Console"
        elif system == "Darwin":
            if os.environ.get("TERM_PROGRAM") == "iTerm.app":
                return "iTerm"
            elif os.environ.get("TERM_PROGRAM") == "Apple_Terminal":
                return "macOS Terminal"
            return "macOS Terminal"
        else:  # Linux or other Unix-like
            if os.environ.get("TERM") is not None:
                if "xterm" in os.environ["TERM"]:
                    return "XTerm"
                elif "screen" in os.environ["TERM"]:
                    return "Screen"
                elif "tmux" in os.environ["TERM"]:
                    return "Tmux"
            return "Unix Terminal"

    def _is_terminal(self) -> bool:
        """
        Check if stdout is attached to a terminal.

        Returns:
            True if stdout is a terminal, False otherwise
        """
        try:
            return sys.stdout.isatty()
        except (AttributeError, OSError):
            return False

    def _detect_windows_legacy_support(self) -> bool:
        """
        Detect color support in legacy Windows consoles.

        Returns:
            True if color is supported, False otherwise
        """
        if platform.system() != "Windows":
            return False

        try:
            # Use this instead of direct test for colorama which might not be installed
            return os.environ.get("ANSICON") is not None
        except Exception:
            return False

    def supports_color(self) -> bool:
        """
        Check if the terminal supports color output.

        Returns:
            True if color is supported, False otherwise
        """
        return (
            self._capabilities["color_support"] and not self._capabilities["no_color"]
        )

    def supports_truecolor(self) -> bool:
        """
        Check if the terminal supports 24-bit true color.

        Returns:
            True if true color is supported, False otherwise
        """
        return self._capabilities["truecolor_support"] and self.supports_color()

    def supports_256color(self) -> bool:
        """
        Check if the terminal supports 256 colors.

        Returns:
            True if 256 colors are supported, False otherwise
        """
        return self._capabilities["256color_support"] and self.supports_color()

    def get_color_depth(self) -> int:
        """
        Get the maximum color depth supported by the terminal.

        Returns:
            One of NO_COLOR, ANSI_16_COLOR, ANSI_256_COLOR, or TRUECOLOR
        """
        if not self.supports_color():
            return self.NO_COLOR
        if self.supports_truecolor():
            return self.TRUECOLOR
        if self.supports_256color():
            return self.ANSI_256_COLOR
        return self.ANSI_16_COLOR

    def get_terminal_type(self) -> str:
        """
        Get the detected terminal type.

        Returns:
            String identifying the terminal type
        """
        return self._terminal_type


class ColorConverter:
    """
    Convert between different color formats based on terminal capabilities.

    This class follows Eidosian principles of flow and precision,
    ensuring optimal visual results across different terminals.
    """

    # Basic ANSI colors (0-7)
    BASIC_COLORS = {
        0: "30",  # Black
        1: "31",  # Red
        2: "32",  # Green
        3: "33",  # Yellow
        4: "34",  # Blue
        5: "35",  # Magenta
        6: "36",  # Cyan
        7: "37",  # White
    }

    # Bright ANSI colors (8-15)
    BRIGHT_COLORS = {
        8: "90",  # Bright Black
        9: "91",  # Bright Red
        10: "92",  # Bright Green
        11: "93",  # Bright Yellow
        12: "94",  # Bright Blue
        13: "95",  # Bright Magenta
        14: "96",  # Bright Cyan
        15: "97",  # Bright White
    }

    @classmethod
    def rgb_to_ansi256(cls, r: int, g: int, b: int) -> str:
        """
        Convert RGB color to ANSI 256-color code.

        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)

        Returns:
            ANSI 256-color code (38;5;{n} or 48;5;{n})
        """
        if r == g == b:
            # Grayscale
            if r < 8:
                return "0"  # Black
            elif r < 248:
                return str(232 + ((r - 8) // 10))  # Grayscale ramp
            else:
                return "15"  # White

        # Convert to 6x6x6 color cube (16-231)
        r_index = max(0, min(5, r // 43))
        g_index = max(0, min(5, g // 43))
        b_index = max(0, min(5, b // 43))

        return str(16 + (r_index * 36) + (g_index * 6) + b_index)

    @classmethod
    def ansi256_to_ansi16(cls, code: int) -> str:
        """
        Convert ANSI 256-color code to ANSI 16-color code.

        Args:
            code: ANSI 256-color code (0-255)

        Returns:
            ANSI 16-color code
        """
        if code < 16:
            # Direct mapping for the first 16 colors
            return (
                cls.BASIC_COLORS.get(code % 8)
                if code < 8
                else cls.BRIGHT_COLORS.get(code % 8)
            )

        elif code >= 232:
            # Grayscale
            level = (code - 232) // 3
            if level < 5:
                return "30"  # Black
            else:
                return "37"  # White
        else:
            # RGB color from 6x6x6 color cube
            code -= 16
            r = (code // 36) * 51
            g = ((code % 36) // 6) * 51
            b = (code % 6) * 51

            # Determine closest ANSI 16 color
            if r == b and r == g:
                # Grayscale
                if r < 75:
                    return "30"  # Black
                elif r < 128:
                    return "90"  # Bright Black (Gray)
                elif r < 192:
                    return "37"  # White
                else:
                    return "97"  # Bright White

            # Find closest color based on maximum component
            max_component = max(r, g, b)
            if r == max_component and r > g * 2 and r > b * 2:
                return "31" if r < 128 else "91"  # Red
            if g == max_component and g > r * 2 and g > b * 2:
                return "32" if g < 128 else "92"  # Green
            if b == max_component and b > r * 2 and b > g * 2:
                return "34" if b < 128 else "94"  # Blue
            if r == max_component and g == max_component and r > b * 2:
                return "33" if r < 128 else "93"  # Yellow
            if r == max_component and b == max_component and r > g * 2:
                return "35" if r < 128 else "95"  # Magenta
            if g == max_component and b == max_component and g > r * 2:
                return "36" if g < 128 else "96"  # Cyan

            # Default to white if no clear match
            return "37" if (r + g + b) / 3 < 128 else "97"

    @classmethod
    def downgrade_color(cls, color_code: str, target_depth: int) -> str:
        """
        Downgrade color code to match the specified color depth.

        Args:
            color_code: Original ANSI color code
            target_depth: Target color depth (16, 256, or 16777216)

        Returns:
            Downgraded color code
        """
        # Extract existing color format
        ansi_match = re.match(r"\033\[(\d+(?:;\d+)*)m", color_code)
        if not ansi_match:
            return color_code

        params = ansi_match.group(1).split(";")

        # Not a color code we recognize for downgrade
        if len(params) < 1:
            return color_code

        result = []
        i = 0
        while i < len(params):
            # Handle different color formats
            if (
                i + 2 < len(params)
                and params[i] in ("38", "48")
                and params[i + 1] == "2"
            ):
                # True color format: 38;2;r;g;b or 48;2;r;g;b
                if target_depth < ColorDetection.TRUECOLOR:
                    if target_depth >= ColorDetection.ANSI_256_COLOR and i + 5 <= len(
                        params
                    ):
                        # Convert to 256 colors
                        r, g, b = map(int, params[i + 2 : i + 5])
                        ansi256 = cls.rgb_to_ansi256(r, g, b)
                        result.extend([params[i], "5", ansi256])
                    else:
                        # Convert to 16 colors
                        r, g, b = map(int, params[i + 2 : i + 5])
                        ansi256 = cls.rgb_to_ansi256(r, g, b)
                        ansi16 = cls.ansi256_to_ansi16(int(ansi256))
                        result.append(ansi16)
                else:
                    result.extend(params[i : i + 5])
                i += 5

            elif (
                i + 2 < len(params)
                and params[i] in ("38", "48")
                and params[i + 1] == "5"
            ):
                # 256 color format: 38;5;n or 48;5;n
                if target_depth < ColorDetection.ANSI_256_COLOR:
                    ansi16 = cls.ansi256_to_ansi16(int(params[i + 2]))
                    if params[i] == "48":
                        # Background color needs adjustment
                        bg_code = str(int(ansi16) + 10)
                        result.append(bg_code)
                    else:
                        result.append(ansi16)
                else:
                    result.extend(params[i : i + 3])
                i += 3

            else:
                # Pass through other codes unchanged
                result.append(params[i])
                i += 1

        return f"\033[{';'.join(result)}m"


# For backward compatibility
terminal_has_colors = lambda: ColorDetection().supports_color()


"""
Compatibility module for color handling across different environments.

This module provides cross-platform color adjustment utilities to ensure
consistent rendering in various terminal environments, including tools
for color detection, terminal capabilities inspection, and graceful fallbacks.
"""

from typing import Dict, Optional

# Color code mapping
COLOR_MAP = {
    "black": 0,
    "red": 1,
    "green": 2,
    "yellow": 3,
    "blue": 4,
    "magenta": 5,
    "cyan": 6,
    "white": 7,
    "default": 9,
}

# Constants for color handling
ANSI_RESET = "\033[0m"
ANSI_COLOR_PATTERN = re.compile(r"\033\[[0-9;]+m")


class ColourAdjuster:
    """
    Handles color adjustments across different terminal environments.

    This class provides utilities to detect color support, adjust colors
    based on terminal capabilities, and ensure consistent rendering
    across different platforms.
    """

    def __init__(self, force_color: bool = False, no_color: bool = False):
        """
        Initialize the ColourAdjuster.

        Args:
            force_color: Force color output even if terminal doesn't support it
            no_color: Force disable color output
        """
        self.force_color = force_color
        self.no_color = no_color
        self._color_supported = None

    @property
    def color_supported(self) -> bool:
        """
        Determine if the terminal supports color output.

        Returns:
            True if color is supported, False otherwise
        """
        if self._color_supported is not None:
            return self._color_supported

        # Honor NO_COLOR environment variable
        if os.environ.get("NO_COLOR") is not None or self.no_color:
            self._color_supported = False
            return False

        # Force color if requested
        if self.force_color:
            self._color_supported = True
            return True

        # Check if output is redirected
        if not sys.stdout.isatty():
            self._color_supported = False
            return False

        # Platform-specific checks
        if platform.system() == "Windows":
            # Windows 10 build 14931+ supports ANSI colors
            if sys.getwindowsversion().build >= 14931:
                self._color_supported = True
                return True

            # Check for color-supporting terminal emulators
            if os.environ.get("TERM_PROGRAM") in (
                "vscode",
                "conemu",
                "cmder",
                "alacritty",
            ):
                self._color_supported = True
                return True

            self._color_supported = False
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
                self._color_supported = True
                return True

            self._color_supported = False
            return False

    def colorize(
        self,
        text: str,
        fg_color: Optional[str] = None,
        bg_color: Optional[str] = None,
        bold: bool = False,
    ) -> str:
        """
        Add color to text based on terminal capabilities.

        Args:
            text: Text to colorize
            fg_color: Foreground color name
            bg_color: Background color name
            bold: Whether to apply bold formatting

        Returns:
            Colorized text string with ANSI codes if supported
        """
        if not self.color_supported or (not fg_color and not bg_color and not bold):
            return text

        # Build ANSI escape sequence
        codes = []

        # Add bold attribute
        if bold:
            codes.append("1")

        # Add foreground color
        if fg_color and fg_color.lower() in COLOR_MAP:
            codes.append(f"3{COLOR_MAP[fg_color.lower()]}")

        # Add background color
        if bg_color and bg_color.lower() in COLOR_MAP:
            codes.append(f"4{COLOR_MAP[bg_color.lower()]}")

        # Combine into ANSI sequence
        if not codes:
            return text

        ansi_code = f'\033[{";".join(codes)}m'
        return f"{ansi_code}{text}{ANSI_RESET}"

    def strip_colors(self, text: str) -> str:
        """
        Remove ANSI color codes from text.

        Args:
            text: Text to process

        Returns:
            Text with all ANSI color codes removed
        """
        return ANSI_COLOR_PATTERN.sub("", text)

    @staticmethod
    def get_color_code(color_name: str, is_background: bool = False) -> str:
        """
        Get ANSI color code for a named color.

        Args:
            color_name: Color name (e.g., 'red', 'blue')
            is_background: Whether this is for background color

        Returns:
            ANSI color code string
        """
        color_name = color_name.lower()
        if color_name not in COLOR_MAP:
            return ""

        base = "4" if is_background else "3"
        return f"\033[{base}{COLOR_MAP[color_name]}m"

    def adapt_for_environment(self, text: str) -> str:
        """
        Adapt colored text for the current terminal environment.

        This ensures color codes are only included when supported and
        handles any platform-specific adjustments needed.

        Args:
            text: Text to adapt

        Returns:
            Environment-appropriate text
        """
        if not self.color_supported:
            return self.strip_colors(text)

        # Windows CMD.EXE before Windows 10 doesn't support ANSI
        if platform.system() == "Windows" and not self.force_color:
            win_ver = sys.getwindowsversion()
            if win_ver.major < 10:
                return self.strip_colors(text)

        return text


# For backwards compatibility with code using original module
def supports_color() -> bool:
    """
    Check if the current terminal supports color output.

    Returns:
        True if color output is supported, False otherwise
    """
    adjuster = ColourAdjuster()
    return adjuster.color_supported


def colorize(text: str, color: str) -> str:
    """
    Apply color to text if the terminal supports it.

    Args:
        text: Text to colorize
        color: Color name

    Returns:
        Colorized text if supported, original text otherwise
    """
    adjuster = ColourAdjuster()
    return adjuster.colorize(text, fg_color=color)


"""
Color adjustment and compatibility module for terminal environments.

Provides tools to detect color capabilities and adapt rendered output to match
terminal color support through elegant downgrading algorithms.
"""

import re
from typing import Dict, Optional

# Singleton instance of ColourAdjuster for module-level functions
colour = None


class ColourAdjuster:
    """
    Detects terminal color capabilities and adapts output accordingly.

    This class follows Eidosian principles of contextual integrity by
    adapting color output to the actual capabilities of the terminal
    rather than assuming capabilities.
    """

    def __init__(self):
        """Initialize the color adjuster with auto-detected capabilities."""
        self._force_mode = self._get_force_mode()
        self._supports_color = self._detect_terminal_color_support()
        self._color_depth = self._detect_color_depth()

        # Cache common regex patterns for performance
        self._ansi_color_regex = re.compile(r"\033\[[0-9;]*m")
        self._truecolor_regex = re.compile(r"\033\[38;2;(\d+);(\d+);(\d+)m")
        self._256color_regex = re.compile(r"\033\[38;5;(\d+)m")
        self._16color_regex = re.compile(r"\033\[(3[0-7])(;1)?m")

        # Color conversion tables
        self._basic_ansi_colors = {
            # Color name: (ANSI code, bright version)
            "black": ("30", "30;1"),
            "red": ("31", "31;1"),
            "green": ("32", "32;1"),
            "yellow": ("33", "33;1"),
            "blue": ("34", "34;1"),
            "magenta": ("35", "35;1"),
            "cyan": ("36", "36;1"),
            "white": ("37", "37;1"),
        }

    def _get_force_mode(self) -> Optional[str]:
        """
        Check environment variables for forced color modes.

        Returns:
            Optional[str]: Forced color mode or None if not forced
        """
        if (
            os.environ.get("NO_COLOR") is not None
            or os.environ.get("FORCE_COLOR") == "0"
        ):
            return "no-color"

        force_color = os.environ.get("FORCE_COLOR")
        if force_color == "1":
            return "16"
        elif force_color == "2":
            return "256"
        elif force_color == "3":
            return "truecolor"

        # Check for truecolor terminal
        if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
            return "truecolor"

        return None

    def _detect_terminal_color_support(self) -> bool:
        """
        Detect if the terminal supports color output.

        Returns:
            bool: True if color is supported, False otherwise
        """
        # Return False if not a TTY
        if not sys.stdout.isatty():
            return False

        # Platform specific checks
        platform = sys.platform
        if platform == "win32":
            # Windows detection logic
            return self._detect_windows_color_support()
        else:
            # Unix-like systems
            return self._detect_unix_color_support()

    def _detect_windows_color_support(self) -> bool:
        """Detect color support on Windows systems."""
        # Modern Windows terminals support colors
        if hasattr(sys, "getwindowsversion") and sys.getwindowsversion().major >= 10:
            return True

        # Check for known color-supporting terminal emulators
        if os.environ.get("TERM_PROGRAM") in ("vscode", "terminal", "alacritty"):
            return True

        # Windows Terminal or ConEmu
        if os.environ.get("WT_SESSION") or os.environ.get("ConEmuANSI"):
            return True

        return False

    def _detect_unix_color_support(self) -> bool:
        """Detect color support on Unix-like systems."""
        # Check TERM environment variable
        term = os.environ.get("TERM", "")
        if term == "dumb":
            return False

        # Common color-supporting terminals
        color_terms = (
            "xterm",
            "xterm-color",
            "xterm-256color",
            "screen",
            "screen-256color",
            "linux",
            "cygwin",
            "ansi",
        )

        return term in color_terms or any(t in term for t in color_terms)

    def _detect_color_depth(self) -> int:
        """
        Detect the color depth supported by the terminal.

        Returns:
            int: 0 (no color), 16, 256, or 16777216 (truecolor)
        """
        if not self._supports_color:
            return 0

        # Check for truecolor support
        if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
            return 16777216

        # Check for 256 color support
        term = os.environ.get("TERM", "")
        if "256color" in term:
            return 256

        # Default to 16 colors
        return 16

    @property
    def supports_color(self) -> bool:
        """Whether the terminal supports color output."""
        if self._force_mode == "no-color":
            return False
        elif self._force_mode in ("16", "256", "truecolor"):
            return True
        return self._supports_color

    @property
    def effective_color_depth(self) -> int:
        """The effective color depth after considering force modes."""
        if not self.supports_color:
            return 0

        if self._force_mode == "16":
            return 16
        elif self._force_mode == "256":
            return 256
        elif self._force_mode == "truecolor":
            return 16777216

        return self._color_depth

    def strip_colors(self, text: str) -> str:
        """
        Remove all ANSI color codes from text.

        Args:
            text: Text containing ANSI color codes

        Returns:
            Clean text without color codes
        """
        return self._ansi_color_regex.sub("", text)

    def downgrade_colors(self, text: str) -> str:
        """
        Adapt color codes to the terminal's capabilities.

        Args:
            text: Text with color codes

        Returns:
            Text with adapted color codes
        """
        if not self.supports_color:
            return self.strip_colors(text)

        depth = self.effective_color_depth

        # True color -> 256 color
        if depth < 16777216:
            text = self._downgrade_truecolor_to_256(text)

        # 256 color -> 16 color
        if depth < 256:
            text = self._downgrade_256_to_16(text)

        return text

    def _downgrade_truecolor_to_256(self, text: str) -> str:
        """Convert 24-bit color codes to 256 color codes."""

        def replace_truecolor(match):
            r, g, b = map(int, match.groups())
            color_256 = self._rgb_to_256(r, g, b)
            return f"\033[38;5;{color_256}m"

        return self._truecolor_regex.sub(replace_truecolor, text)

    def _downgrade_256_to_16(self, text: str) -> str:
        """Convert 256 color codes to 16 color codes."""

        def replace_256color(match):
            color = int(match.group(1))
            ansi_code = self._256_to_16(color, "3")
            return f"\033[{ansi_code}m"

        return self._256color_regex.sub(replace_256color, text)

    def _rgb_to_256(self, r: int, g: int, b: int) -> int:
        """
        Convert RGB values to the nearest 256-color code.

        Args:
            r, g, b: RGB values (0-255)

        Returns:
            256-color code (16-231 for colors, 232-255 for grays)
        """
        # Handle grayscale (equal RGB values)
        if r == g == b:
            if r < 8:
                return 16  # black
            if r > 248:
                return 231  # white
            # Calculate grayscale index (232-255)
            return 232 + ((r - 8) // 10)

        # Calculate indices in the 6x6x6 RGB color cube (16-231)
        r_index = max(0, min(5, r // 43))
        g_index = max(0, min(5, g // 43))
        b_index = max(0, min(5, b // 43))

        return 16 + (36 * r_index) + (6 * g_index) + b_index

    def _256_to_16(self, color: int, base_code: str) -> str:
        """
        Convert 256-color code to 16-color ANSI code.

        Args:
            color: 256-color code (0-255)
            base_code: '3' for foreground, '4' for background

        Returns:
            ANSI color code
        """
        # Standard ANSI colors (0-7)
        if color < 8:
            return f"{base_code}{color}"

        # Bright standard colors (8-15)
        if color < 16:
            return f"{base_code}{color - 8};1"

        # 6x6x6 color cube (16-231)
        if color < 232:
            # Extract r,g,b indices from the cube
            color -= 16
            r = color // 36
            g = (color % 36) // 6
            b = color % 6

            # Get weighted RGB values
            r_val = 0 if r == 0 else (r * 40 + 55)
            g_val = 0 if g == 0 else (g * 40 + 55)
            b_val = 0 if b == 0 else (b * 40 + 55)

            # Convert to basic ANSI color
            return self._rgb_to_16(r_val, g_val, b_val, base_code)

        # Grayscale (232-255)
        gray = ((color - 232) * 10) + 8
        # Handle as RGB for consistency
        return self._rgb_to_16(gray, gray, gray, base_code)

    def _rgb_to_16(self, r: int, g: int, b: int, base_code: str) -> str:
        """
        Convert RGB to 16-color ANSI code.

        This uses a simplified algorithm to determine the closest ANSI color.

        Args:
            r, g, b: RGB values (0-255)
            base_code: '3' for foreground, '4' for background

        Returns:
            ANSI color code
        """
        # Determine intensity (bright or normal)
        bright = r > 170 or g > 170 or b > 170

        # Find dominant color
        if r > g and r > b:
            color = "1"  # red
        elif g > r and g > b:
            color = "2"  # green
        elif b > r and b > g:
            color = "4"  # blue
        elif r > 170 and g > 170:
            color = "3"  # yellow
        elif g > 170 and b > 170:
            color = "6"  # cyan
        elif r > 170 and b > 170:
            color = "5"  # magenta
        elif r > 170 or g > 170 or b > 170:
            color = "7"  # white
        else:
            color = "0"  # black

        # Build the ANSI code
        if bright and color != "0":
            return f"{base_code}{color};1"
        return f"{base_code}{color}"


def adapt_colors_to_terminal(text: str) -> str:
    """
    Convert color codes in text to match terminal capabilities.

    Args:
        text: Text with color codes

    Returns:
        Text with colors adapted to terminal capabilities
    """
    global colour
    if colour is None:
        colour = ColourAdjuster()
    return colour.downgrade_colors(text)


def strip_colors(text: str) -> str:
    """
    Remove all color codes from text.

    Args:
        text: Text with color codes

    Returns:
        Text without color codes
    """
    global colour
    if colour is None:
        colour = ColourAdjuster()
    return colour.strip_colors(text)


def supports_color() -> bool:
    """
    Check if the terminal supports color output.

    Returns:
        True if color is supported, False otherwise
    """
    global colour
    if colour is None:
        colour = ColourAdjuster()
    return colour.supports_color


# Initialize the singleton
colour = ColourAdjuster()
