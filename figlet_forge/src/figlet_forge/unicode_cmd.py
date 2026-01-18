"""
Unicode System Command Interface for Figlet Forge.

This module provides a comprehensive interface to the system unicode command,
enabling advanced character lookups, transformations, and information retrieval
according to Eidosian principles of clarity, functionality, and maintainability.
"""

import shutil
import subprocess
from enum import Enum
from typing import List, Optional, Union

# Define constants
DEFAULT_CHARSET = "UTF-8"
DEFAULT_MAX_COUNT = 10


class UnicodeInputMode(Enum):
    """Input modes for the unicode command."""

    AUTO = "auto"
    HEXADECIMAL = "hexadecimal"
    OCTAL = "octal"
    BINARY = "binary"
    DECIMAL = "decimal"
    REGEXP = "regexp"
    STRING = "string"


class UnicodeCommand:
    """
    Interface to the system unicode command with comprehensive option support.

    This class provides a Pythonic interface to all functionality of the system
    unicode command, supporting various input formats, display options, and
    output formatting capabilities.
    """

    def __init__(self):
        """Initialize the UnicodeCommand interface."""
        self._command_path = shutil.which("unicode")
        if self._command_path is None:
            raise FileNotFoundError(
                "Unicode command not found. Please ensure it is installed and in PATH."
            )

    def lookup(
        self,
        query: str,
        mode: Union[UnicodeInputMode, str] = UnicodeInputMode.AUTO,
        max_count: int = DEFAULT_MAX_COUNT,
        io_charset: str = DEFAULT_CHARSET,
        from_cp: Optional[str] = None,
        add_charset: Optional[str] = None,
        use_color: Optional[str] = None,
        verbose: bool = False,
        wikipedia: bool = False,
        wiktionary: bool = False,
        brief: bool = False,
        format_string: Optional[str] = None,
    ) -> str:
        """
        Look up Unicode information using the system unicode command.

        Args:
            query: The character, codepoint, or pattern to look up
            mode: Input interpretation mode (auto, hexadecimal, etc.)
            max_count: Maximum number of codepoints to display (0=unlimited)
            io_charset: I/O character set
            from_cp: Convert numerical arguments from this encoding
            add_charset: Show hexadecimal representation in additional charset
            use_color: Whether to use colors ('on', 'off', 'auto')
            verbose: Increase verbosity (reads Unihan properties)
            wikipedia: Query Wikipedia for the character
            wiktionary: Query Wiktionary for the character
            brief: Use brief output format
            format_string: Custom formatting string

        Returns:
            Output from the unicode command as a string

        Raises:
            subprocess.SubprocessError: If the unicode command fails
            FileNotFoundError: If the unicode command is not found
        """
        # Build command arguments
        args = [self._command_path]

        # Process mode
        if isinstance(mode, UnicodeInputMode):
            mode = mode.value

        if mode == UnicodeInputMode.HEXADECIMAL.value:
            args.append("--hexadecimal")
        elif mode == UnicodeInputMode.OCTAL.value:
            args.append("--octal")
        elif mode == UnicodeInputMode.BINARY.value:
            args.append("--binary")
        elif mode == UnicodeInputMode.DECIMAL.value:
            args.append("--decimal")
        elif mode == UnicodeInputMode.REGEXP.value:
            args.append("--regexp")
        elif mode == UnicodeInputMode.STRING.value:
            args.append("--string")
        elif mode == UnicodeInputMode.AUTO.value:
            args.append("--auto")

        # Add other options
        if max_count != DEFAULT_MAX_COUNT:
            args.extend(["--max", str(max_count)])

        if io_charset != DEFAULT_CHARSET:
            args.extend(["--io", io_charset])

        if from_cp:
            args.extend(["--fromcp", from_cp])

        if add_charset:
            args.extend(["--charset-add", add_charset])

        if use_color:
            args.extend(["--colour", use_color])

        if verbose:
            args.append("--verbose")

        if wikipedia:
            args.append("--wikipedia")

        if wiktionary:
            args.append("--wiktionary")

        if brief:
            args.append("--brief")

        if format_string:
            args.extend(["--format", format_string])

        # Add the query
        args.append(query)

        # Execute the command
        try:
            result = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            if e.stderr:
                return f"Unicode command error: {e.stderr}"
            return f"Unicode command failed with code {e.returncode}"

    def list_encodings(self) -> List[str]:
        """
        List all known character encodings.

        Returns:
            List of available character encodings
        """
        try:
            result = subprocess.run(
                [self._command_path, "--list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            # Parse the output to get the list of encodings
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except subprocess.CalledProcessError as e:
            if e.stderr:
                raise RuntimeError(f"Unicode command error: {e.stderr}")
            raise RuntimeError(f"Unicode command failed with code {e.returncode}")

    def display_ascii_table(self, brexit: bool = False) -> str:
        """
        Display ASCII table.

        Args:
            brexit: Whether to use the EU-UK Trade and Cooperation Agreement version

        Returns:
            ASCII table as a string
        """
        args = [self._command_path]

        if brexit:
            args.append("--brexit-ascii")
        else:
            args.append("--ascii")

        try:
            result = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            if e.stderr:
                return f"Unicode command error: {e.stderr}"
            return f"Unicode command failed with code {e.returncode}"

    def download_unicode_data(self) -> bool:
        """
        Download UnicodeData.txt.

        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                [self._command_path, "--download"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return "downloaded successfully" in result.stdout.lower()
        except subprocess.CalledProcessError:
            return False


def get_unicode_info(
    char: str, mode: UnicodeInputMode = UnicodeInputMode.STRING, verbose: bool = False
) -> str:
    """
    Get Unicode information for a character or codepoint.

    Convenience function for quick character lookups.

    Args:
        char: Character or codepoint to look up
        mode: Input interpretation mode
        verbose: Whether to show verbose information

    Returns:
        Unicode information as a string

    Raises:
        FileNotFoundError: If the unicode command is not available
    """
    try:
        cmd = UnicodeCommand()
        return cmd.lookup(char, mode=mode, verbose=verbose)
    except FileNotFoundError:
        return f"System unicode command not available. Character: {char}"


# Test if unicode command is available on the system
def is_available() -> bool:
    """
    Check if the system unicode command is available.

    Returns:
        True if the unicode command is available, False otherwise
    """
    return shutil.which("unicode") is not None


"""
Unicode command handling for Figlet Forge.

This module provides utilities for processing Unicode characters in
figlet text, ensuring proper rendering across different terminals
and environments.
"""

import sys
import unicodedata
from typing import List, Optional, Union


def is_unicode_supported() -> bool:
    """
    Check if the current environment supports Unicode output.

    Returns:
        True if Unicode is likely supported, False otherwise
    """
    try:
        # Check if stdout can handle Unicode
        encoding = sys.stdout.encoding or "ascii"
        return encoding.lower() in ("utf-8", "utf8", "utf_8")
    except (AttributeError, Exception):
        # Default to assume support on modern systems
        return True


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text for consistent rendering.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text using NFC form
    """
    try:
        return unicodedata.normalize("NFC", text)
    except Exception:
        # Return original if normalization fails
        return text


def get_char_width(char: str) -> int:
    """
    Get the display width of a Unicode character.

    Handles full-width and combining characters appropriately.

    Args:
        char: Unicode character to measure

    Returns:
        Display width in terminal columns (0, 1, or 2)
    """
    if not char or char == "":
        return 0

    # Handle common control and zero-width characters
    if ord(char) < 32 or unicodedata.category(char).startswith("C"):
        return 0

    # Handle combining characters
    if unicodedata.combining(char) > 0:
        return 0

    # Handle wide characters (CJK, etc.)
    try:
        east_asian_width = unicodedata.east_asian_width(char)
        if east_asian_width in ("F", "W"):  # Full-width or Wide
            return 2
    except Exception:
        pass

    # Default to standard width
    return 1


def measure_string_width(text: str) -> int:
    """
    Measure the display width of a string with Unicode awareness.

    Args:
        text: String to measure

    Returns:
        Display width in terminal columns
    """
    if not text:
        return 0

    width = 0
    for char in text:
        width += get_char_width(char)
    return width


def truncate_to_width(text: str, width: int) -> str:
    """
    Truncate string to specified display width with Unicode awareness.

    Args:
        text: String to truncate
        width: Maximum display width

    Returns:
        Truncated string fitting within width
    """
    if not text:
        return ""

    result = []
    current_width = 0

    for char in text:
        char_width = get_char_width(char)
        if current_width + char_width > width:
            break
        result.append(char)
        current_width += char_width

    return "".join(result)


def is_full_width_char(char: str) -> bool:
    """
    Check if a character is full-width (takes 2 columns in terminal).

    Args:
        char: Character to check

    Returns:
        True if character is full-width, False otherwise
    """
    if not char or len(char) != 1:
        return False

    try:
        east_asian_width = unicodedata.east_asian_width(char)
        return east_asian_width in ("F", "W")  # Full-width or Wide
    except Exception:
        return False


def get_unicode_block(char: str) -> str:
    """
    Determine which Unicode block a character belongs to.

    Args:
        char: Unicode character to check

    Returns:
        Name of the Unicode block, or "Unknown" if not determinable
    """
    if not char or len(char) != 1:
        return "Unknown"

    code = ord(char)
    # Define common Unicode blocks
    blocks = [
        (0x0000, 0x007F, "Basic Latin"),
        (0x0080, 0x00FF, "Latin-1 Supplement"),
        (0x0100, 0x017F, "Latin Extended-A"),
        (0x0180, 0x024F, "Latin Extended-B"),
        (0x0250, 0x02AF, "IPA Extensions"),
        (0x02B0, 0x02FF, "Spacing Modifier Letters"),
        (0x0300, 0x036F, "Combining Diacritical Marks"),
        (0x0370, 0x03FF, "Greek and Coptic"),
        (0x0400, 0x04FF, "Cyrillic"),
        (0x0500, 0x052F, "Cyrillic Supplement"),
        (0x0530, 0x058F, "Armenian"),
        (0x0590, 0x05FF, "Hebrew"),
        (0x0600, 0x06FF, "Arabic"),
        (0x0700, 0x074F, "Syriac"),
        (0x0750, 0x077F, "Arabic Supplement"),
        # CJK blocks
        (0x3000, 0x303F, "CJK Symbols and Punctuation"),
        (0x3040, 0x309F, "Hiragana"),
        (0x30A0, 0x30FF, "Katakana"),
        (0x3100, 0x312F, "Bopomofo"),
        (0x4E00, 0x9FFF, "CJK Unified Ideographs"),
        (0xFF00, 0xFFEF, "Halfwidth and Fullwidth Forms"),
    ]

    for start, end, name in blocks:
        if start <= code <= end:
            return name

    return "Other"
