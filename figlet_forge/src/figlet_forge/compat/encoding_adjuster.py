from eidosian_core import eidosian
"""
Cross-platform encoding utilities for Figlet Forge.

This module provides functions to handle encoding issues across different platforms,
ensuring text is properly encoded and decoded regardless of the environment.
"""

import locale
import os
import platform
import sys
from typing import Dict, List, Optional, Union


@eidosian()
def supports_utf8() -> bool:
    """
    Check if the current environment supports UTF-8 encoding.

    Returns:
        True if UTF-8 is supported, False otherwise
    """
    try:
        # Check system encoding
        system_encoding = sys.getdefaultencoding()
        if system_encoding.lower() == "utf-8":
            return True

        # Check locale settings
        loc = locale.getlocale()
        if loc[1] and "utf" in loc[1].lower():
            return True

        # Check environment variables
        for env_var in ("LC_ALL", "LC_CTYPE", "LANG"):
            if env_var in os.environ and "utf" in os.environ[env_var].lower():
                return True

        return False
    except Exception:
        # Default to safe value if we can't determine
        return False


@eidosian()
def decode_bytes(byte_data: bytes, encoding: Optional[str] = None) -> str:
    """
    Decode bytes to string using the specified or detected encoding.

    Args:
        byte_data: The bytes to decode
        encoding: Optional encoding to use

    Returns:
        Decoded string
    """
    if not byte_data:
        return ""

    if encoding is None:
        # Try to detect encoding
        encoding = detect_encoding()

    # Try specified encoding first
    try:
        return byte_data.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        # Fallback to utf-8
        try:
            return byte_data.decode("utf-8")
        except UnicodeDecodeError:
            # Final fallback
            return byte_data.decode("latin-1", errors="replace")


@eidosian()
def encode_text(text: str, encoding: Optional[str] = None) -> bytes:
    """
    Encode text to bytes using the specified or default encoding.

    Args:
        text: The string to encode
        encoding: Optional encoding to use

    Returns:
        Encoded bytes
    """
    if not text:
        return b""

    if encoding is None:
        # Try to detect encoding
        encoding = detect_encoding()

    # Try specified encoding first
    try:
        return text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        # Fallback to utf-8
        try:
            return text.encode("utf-8")
        except UnicodeEncodeError:
            # Final fallback with replacement for problematic characters
            return text.encode("utf-8", errors="replace")


@eidosian()
def detect_encoding() -> str:
    """
    Attempt to detect the appropriate encoding for the current environment.

    Returns:
        Detected encoding name
    """
    # Start with system default encoding
    encoding = sys.getdefaultencoding()

    # Check if we should use UTF-8
    if supports_utf8():
        return "utf-8"

    # Check locale
    try:
        loc_encoding = locale.getpreferredencoding()
        if loc_encoding:
            return loc_encoding
    except (locale.Error, AttributeError):
        pass

    # Fallback
    return encoding or "utf-8"


@eidosian()
def normalize_newlines(text: Union[str, bytes]) -> Union[str, bytes]:
    """
    Normalize different newline styles to the platform's default.

    Args:
        text: Input text or bytes with potentially mixed newline styles

    Returns:
        Text or bytes with normalized newlines
    """
    if not text:
        return text

    is_bytes = isinstance(text, bytes)

    if is_bytes:
        # Convert to string temporarily
        text_str = text.decode("latin-1")
    else:
        text_str = text

    # Normalize to \n first
    normalized = text_str.replace("\r\n", "\n").replace("\r", "\n")

    # Then convert to platform-specific newlines if needed
    if os.linesep != "\n":
        normalized = normalized.replace("\n", os.linesep)

    # Convert back to bytes if input was bytes
    if is_bytes:
        return normalized.encode("latin-1")

    return normalized


class EncodingAdjuster:
    """
    Detect and adjust text encoding for terminals.

    This class follows Eidosian principles of contextual awareness,
    ensuring robust text rendering across platforms.
    """

    def __init__(self):
        """Initialize encoding detector with system capabilities."""
        self._encoding_info = self._detect_encoding()

    def _detect_encoding(self) -> Dict[str, any]:
        """
        Detect terminal and system encoding capabilities.

        Returns:
            Dictionary of encoding information
        """
        info = {
            "system_encoding": sys.getdefaultencoding(),
            "locale_encoding": locale.getpreferredencoding(False),
            "stdout_encoding": self._get_stdout_encoding(),
            "supports_utf8": self._supports_utf8(),
            "normalize_needed": platform.system() == "Windows",
            "locale_info": locale.getlocale(),
        }

        return info

    def _get_stdout_encoding(self) -> str:
        """
        Get the encoding used by stdout.

        Returns:
            Encoding name or 'ascii' if unknown
        """
        try:
            return sys.stdout.encoding or "ascii"
        except (AttributeError, ValueError):
            return "ascii"

    def _supports_utf8(self) -> bool:
        """
        Check if the terminal supports UTF-8 output.

        Returns:
            True if UTF-8 is supported, False otherwise
        """
        # Check if stdout encoding supports UTF-8
        stdout_enc = self._get_stdout_encoding().lower()
        if "utf" in stdout_enc:
            return True

        # Check if locale supports UTF-8
        try:
            lang = os.environ.get("LANG", "").lower()
            if "utf" in lang:
                return True
        except Exception:
            pass

        # Check if Windows Terminal/ConPTY supports UTF-8
        if platform.system() == "Windows":
            # Modern Windows Terminal and Windows 10 ConPTY support UTF-8
            if "WT_SESSION" in os.environ or "TERM_PROGRAM" in os.environ:
                return True

            try:
                # Windows 10 1903+ can support UTF-8 output
                version = sys.getwindowsversion()
                if version.major >= 10 and version.build >= 18362:
                    return True
            except (AttributeError, ValueError):
                pass

        return False

    @eidosian()
    def adjust_for_output(self, text: str) -> str:
        """
        Adjust text for terminal output based on encoding capabilities.

        Args:
            text: Text to adjust

        Returns:
            Adjusted text ready for terminal output
        """
        if not text:
            return text

        # If UTF-8 is supported, no adjustment needed
        if self._encoding_info["supports_utf8"]:
            return text

        # Try to encode and decode with terminal encoding
        try:
            encoding = self._encoding_info["stdout_encoding"]
            return text.encode(encoding, errors="replace").decode(encoding)
        except (UnicodeError, LookupError):
            # Fallback to ASCII with replacement
            return text.encode("ascii", errors="replace").decode("ascii")

    @eidosian()
    def encode_for_file(self, text: str, encoding: Optional[str] = None) -> bytes:
        """
        Encode text for file output.

        Args:
            text: Text to encode
            encoding: Target encoding (defaults to system encoding)

        Returns:
            Encoded bytes
        """
        target_encoding = encoding or self._encoding_info["system_encoding"]

        try:
            return text.encode(target_encoding, errors="replace")
        except (UnicodeError, LookupError):
            # Fallback to UTF-8
            return text.encode("utf-8", errors="replace")

    @eidosian()
    def get_encoding_info(self) -> Dict[str, any]:
        """
        Get detected encoding information.

        Returns:
            Dictionary of encoding information
        """
        return self._encoding_info

    @eidosian()
    def normalize_newlines(self, text: str) -> str:
        """
        Normalize newlines for the current platform.

        Args:
            text: Text with potentially mixed newlines

        Returns:
            Text with platform-appropriate newlines
        """
        if not text:
            return text

        # Convert all newlines to \n first
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # Then convert to platform-specific newlines if needed
        if os.linesep != "\n" and self._encoding_info["normalize_needed"]:
            normalized = normalized.replace("\n", os.linesep)

        return normalized

    @eidosian()
    def strip_non_ascii(self, text: str) -> str:
        """
        Strip non-ASCII characters from text.

        Args:
            text: Text to process

        Returns:
            ASCII-only text
        """
        return "".join(c for c in text if ord(c) < 128)

    @eidosian()
    def decode_bytes(
        self, data: Union[bytes, str], encodings: Optional[List[str]] = None
    ) -> str:
        """
        Decode bytes to string using multiple encodings if needed.

        This function tries multiple encodings in sequence to ensure
        successful decoding, with graceful fallback options.

        Args:
            data: Bytes data to decode, or string to ensure is valid
            encodings: List of encodings to try, in order of preference

        Returns:
            Decoded string

        Raises:
            UnicodeError: If decoding fails with all attempted encodings
        """
        # If already a string, just return it
        if isinstance(data, str):
            return data

        # Default encodings to try
        if encodings is None:
            encodings = [
                self._encoding_info["system_encoding"],
                "utf-8",
                "latin-1",
                "ascii",
            ]

        # Try each encoding in order
        for encoding in encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue

        # Last resort - use replace error handler with the first encoding
        return data.decode(encodings[0], errors="replace")


# Create a singleton instance for easy access
_encoding_adjuster = None


@eidosian()
def get_encoding_adjuster() -> EncodingAdjuster:
    """Get the singleton EncodingAdjuster instance."""
    global _encoding_adjuster
    if _encoding_adjuster is None:
        _encoding_adjuster = EncodingAdjuster()
    return _encoding_adjuster


# Module-level convenience functions
@eidosian()
def get_system_encoding() -> str:
    """
    Get the current system encoding.

    Returns:
        System encoding string
    """
    global encoding
    if encoding is None:
        encoding = EncodingAdjuster()
    return encoding.encoding


@eidosian()
def supports_utf8() -> bool:
    """
    Check if the system supports UTF-8 encoding.

    Returns:
        True if UTF-8 is supported
    """
    global encoding
    if encoding is None:
        encoding = EncodingAdjuster()
    return encoding.supports_utf8


@eidosian()
def decode_bytes(data: bytes) -> str:
    """
    Decode bytes to string using system encoding.

    Args:
        data: Bytes to decode

    Returns:
        Decoded string
    """
    global encoding
    if encoding is None:
        encoding = EncodingAdjuster()
    return encoding.decode(data)


@eidosian()
def encode_text(text: str) -> bytes:
    """
    Encode string to bytes using system encoding.

    Args:
        text: String to encode

    Returns:
        Encoded bytes
    """
    global encoding
    if encoding is None:
        encoding = EncodingAdjuster()
    return encoding.encode(text)


# Initialize singleton
encoding = EncodingAdjuster()
