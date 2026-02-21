"""
Tests for encoding detection and compatibility handling.

Ensures that encoding detection and adjustment work correctly across
different locales, operating systems, and terminal environments.
"""

import locale
import os
from unittest.mock import MagicMock, patch

import pytest

from figlet_forge.compat.encoding_adjuster import (
    EncodingAdjuster,
    decode_bytes,
    encode_text,
    get_system_encoding,
    supports_utf8,
)


class TestEncodingDetection:
    """Test encoding detection capabilities."""

    @staticmethod
    def _norm_encoding(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    def test_default_init(self):
        """Test default initialization of EncodingAdjuster."""
        adjuster = EncodingAdjuster()
        assert isinstance(adjuster.encoding, str)
        assert adjuster.encoding != ""
        assert isinstance(adjuster.supports_utf8, bool)

    @pytest.mark.parametrize(
        "locale_value,expected_encoding",
        [
            ("en_US.UTF-8", "utf-8"),
            ("C", "ascii"),
            ("POSIX", "ascii"),
            ("en_US.ISO8859-1", "iso-8859-1"),
        ],
    )
    def test_locale_based_detection(self, locale_value, expected_encoding):
        """Test encoding detection based on locale settings."""
        with patch("locale.getpreferredencoding") as mock_locale:
            mock_locale.return_value = expected_encoding

            adjuster = EncodingAdjuster()
            # Case insensitive comparison since system might return different casing
            assert self._norm_encoding(adjuster.encoding) == self._norm_encoding(expected_encoding)

    @pytest.mark.parametrize(
        "env_vars,expected_encoding",
        [
            ({"LC_ALL": "en_US.UTF-8"}, "utf-8"),
            ({"LC_CTYPE": "en_US.UTF-8"}, "utf-8"),
            ({"LANG": "en_US.UTF-8"}, "utf-8"),
            ({"LANG": "en_US.ISO8859-1"}, "iso-8859-1"),
            ({}, "utf-8"),  # Default case
        ],
    )
    def test_environment_based_detection(self, env_vars, expected_encoding):
        """Test encoding detection from environment variables."""
        with patch.dict(os.environ, env_vars, clear=True), patch(
            "locale.getpreferredencoding", side_effect=locale.Error
        ):
            adjuster = EncodingAdjuster()
            assert self._norm_encoding(adjuster.encoding) == self._norm_encoding(expected_encoding)

    def test_stdout_encoding_fallback(self):
        """Test fallback to stdout encoding."""
        with patch("locale.getpreferredencoding", side_effect=locale.Error), patch.dict(
            os.environ, {}, clear=True
        ), patch("sys.stdout") as mock_stdout:
            mock_stdout.encoding = "windows-1252"
            adjuster = EncodingAdjuster()
            assert adjuster.encoding.lower() == "windows-1252".lower()


class TestEncodingOperations:
    """Test encoding and decoding operations."""

    def test_decode_utf8(self):
        """Test decoding UTF-8 bytes."""
        adjuster = EncodingAdjuster()
        text = "Hello 世界"  # Hello World in Chinese
        bytes_data = text.encode("utf-8")
        assert adjuster.decode(bytes_data) == text

    def test_decode_latin1_fallback(self):
        """Test fallback to latin-1 for invalid encodings."""
        adjuster = EncodingAdjuster()
        # These bytes are invalid in UTF-8 but valid in latin-1
        bytes_data = b"\xff\xfe\xfd"

        # Should not raise an error due to fallback
        result = adjuster.decode(bytes_data)
        assert result is not None
        assert len(result) == len(bytes_data)

    def test_encode_with_fallback(self):
        """Test encoding with ASCII fallback for non-ASCII characters."""
        adjuster = EncodingAdjuster()
        text = "Hello 世界"  # Contains non-ASCII

        # Simulate ASCII-only environment
        with patch.object(adjuster, "_encoding", "ascii"):
            # Should not raise but use replacement character
            result = adjuster.encode(text)
            assert result is not None
            assert len(result) >= len("Hello ")  # At least the ASCII part

    def test_ensure_unicode(self):
        """Test ensuring text is Unicode-compatible."""
        adjuster = EncodingAdjuster()
        text = "Hello 世界"

        # Test with UTF-8 support
        with patch.object(adjuster, "_can_encode_utf8", True):
            result = adjuster.ensure_unicode(text)
            assert result == text  # Should remain unchanged

        # Test without UTF-8 support
        with patch.object(adjuster, "_can_encode_utf8", False):
            result = adjuster.ensure_unicode(text)
            assert "Hello" in result  # ASCII part should remain
            assert "世界" not in result  # Non-ASCII should be replaced

    def test_convenience_functions(self):
        """Test the module-level convenience functions."""
        with patch("figlet_forge.compat.encoding_adjuster.encoding") as mock_encoding:
            mock_encoding.encoding = "utf-8"
            mock_encoding.supports_utf8 = True
            mock_encoding.decode.return_value = "decoded"
            mock_encoding.encode.return_value = b"encoded"

            assert get_system_encoding() == "utf-8"
            assert supports_utf8() is True
            assert decode_bytes(b"test") == "decoded"
            assert encode_text("test") == b"encoded"


class TestUTF8Detection:
    """Test detection of UTF-8 support."""

    def test_detect_utf8_support(self):
        """Test detection of UTF-8 decoding support."""
        adjuster = EncodingAdjuster()

        # Test with UTF-8 compatible encoding
        with patch.object(adjuster, "_encoding", "utf-8"):
            result = adjuster._check_utf8_support()
            assert result is True

        # Test with incompatible encoding
        with patch.object(adjuster, "_encoding", "ascii"):
            result = adjuster._check_utf8_support()
            assert result is False

    def test_detect_utf8_output(self):
        """Test detection of UTF-8 output support."""
        adjuster = EncodingAdjuster()

        # Test with TTY and UTF-8 encoding
        stdout_mock = MagicMock()
        stdout_mock.isatty.return_value = True
        stdout_mock.encoding = "utf-8"
        with patch("sys.stdout", stdout_mock), patch.dict(os.environ, {}, clear=True):
            result = adjuster._check_utf8_output()
            assert result is True

        # Test with TTY and non-UTF-8 encoding
        stdout_mock = MagicMock()
        stdout_mock.isatty.return_value = True
        stdout_mock.encoding = "ascii"
        with patch("sys.stdout", stdout_mock), patch.dict(os.environ, {}, clear=True):
            result = adjuster._check_utf8_output()
            assert result is False

        # Test without TTY (redirected output)
        with patch("sys.stdout.isatty", return_value=False):
            result = adjuster._check_utf8_output()
            assert result is True  # Assume redirected output can handle UTF-8

        # Test with UTF-8 environment variable
        stdout_mock = MagicMock()
        stdout_mock.isatty.return_value = True
        stdout_mock.encoding = "ascii"
        with patch("sys.stdout", stdout_mock), patch.dict(os.environ, {"LANG": "en_US.UTF-8"}, clear=True):
            result = adjuster._check_utf8_output()
            assert result is True


# Run these tests with different encoding environments if possible
@pytest.mark.skipif(
    "CI" in os.environ or "GITHUB_ACTIONS" in os.environ,
    reason="These tests are sensitive to system configuration",
)
class TestRealEncodingEnvironment:
    """Tests using the real system encoding environment."""

    def test_real_system_encoding(self):
        """Test detection of the real system encoding."""
        real_encoding = locale.getpreferredencoding()
        adjuster = EncodingAdjuster()
        assert adjuster.encoding.lower() == real_encoding.lower()

    def test_real_utf8_support(self):
        """Test detection of real UTF-8 support."""
        adjuster = EncodingAdjuster()
        # Just verify it returns a boolean without errors
        assert isinstance(adjuster.supports_utf8, bool)

    def test_real_encode_decode(self):
        """Test real encoding and decoding operations."""
        adjuster = EncodingAdjuster()
        text = "Hello, World!"
        encoded = adjuster.encode(text)
        decoded = adjuster.decode(encoded)
        assert decoded == text
