"""
Unit tests for utility functions in Figlet Forge.

These tests verify the correctness of helper functions and utilities
used throughout the codebase.
"""

import os
import unittest
from pathlib import Path
from typing import Any, Dict

import pytest

from figlet_forge import Figlet
from figlet_forge.core.utils import (
    get_terminal_size,
    merge_dicts,
    normalize_path,
    safe_read_file,
    strip_ansi_codes,
    unicode_string,
)


# For backward compatibility testing
def figlet_format(text, **kwargs):
    """Compatibility wrapper for the original pyfiglet function."""
    fig = Figlet(**kwargs)
    return fig.renderText(text)


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_unicode_string(self) -> None:
        """Test unicode string handling."""
        # Test with ASCII
        ascii_text = "Hello"
        self.assertEqual(unicode_string(ascii_text), ascii_text)

        # Test with Unicode
        unicode_text = "Hello 世界"
        self.assertEqual(unicode_string(unicode_text), unicode_text)

    def test_normalize_path(self) -> None:
        """Test path normalization."""
        # Test with forward slashes
        path = "path/to/file"
        expected = os.path.normpath(path)
        self.assertEqual(normalize_path(path), expected)

        # Test with backslashes if on Windows
        if os.name == "nt":
            path = r"path\to\file"
            expected = os.path.normpath(path)
            self.assertEqual(normalize_path(path), expected)

    def test_strip_ansi_codes(self) -> None:
        """Test stripping ANSI color codes."""
        # Test with ANSI colors
        colored = "\033[31mRed Text\033[0m"
        self.assertEqual(strip_ansi_codes(colored), "Red Text")

        # Test with multiple codes
        complex_colors = "\033[1;32;44mComplex\033[0m"
        self.assertEqual(strip_ansi_codes(complex_colors), "Complex")

        # Test with no codes
        plain = "Plain text"
        self.assertEqual(strip_ansi_codes(plain), plain)

    def test_merge_dicts(self) -> None:
        """Test dictionary merging."""
        # Test basic merge
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        expected = {"a": 1, "b": 3, "c": 4}
        self.assertEqual(merge_dicts(dict1, dict2), expected)

        # Test with empty dictionaries
        self.assertEqual(merge_dicts({}, dict1), dict1)
        self.assertEqual(merge_dicts(dict1, {}), dict1)

        # Test that original dicts are not modified
        merge_dicts(dict1, dict2)
        self.assertEqual(dict1, {"a": 1, "b": 2})
        self.assertEqual(dict2, {"b": 3, "c": 4})

    def test_figlet_format(self) -> None:
        """Test figlet_format utility function."""
        # Basic usage
        result = figlet_format("Test")
        self.assertTrue(isinstance(result, str))
        self.assertTrue(result)  # Non-empty string

        # With custom font
        try:
            result = figlet_format("Test", font="slant")
            self.assertTrue(isinstance(result, str))
            self.assertTrue(result)  # Non-empty string
        except Exception:
            # Skip if font not available
            pass


@pytest.mark.parametrize(
    "input_text,expected_identical",
    [
        ("Hello", True),  # ASCII text should be unchanged
        ("Hello 世界", True),  # Unicode should be preserved
        ("", True),  # Empty string should remain empty
    ],
)
def test_unicode_string_parametrized(input_text: str, expected_identical: bool) -> None:
    """
    Test unicode_string function with parametrization.

    Args:
        input_text: Input text to process
        expected_identical: Whether output should match input
    """
    result = unicode_string(input_text)
    if expected_identical:
        assert result == input_text
    else:
        assert result != input_text


def test_get_terminal_size(mock_env_vars: Dict[str, str]) -> None:
    """
    Test terminal size detection with environment variable mocking.

    Args:
        mock_env_vars: Environment variables fixture
    """
    # Set test environment variables
    os.environ["COLUMNS"] = "120"
    os.environ["LINES"] = "40"

    # Get terminal size (should use env vars)
    width, height = get_terminal_size()

    # Verify results
    assert width == 120
    assert height == 40

    # Test with invalid values (should fall back to defaults)
    os.environ["COLUMNS"] = "invalid"
    width, height = get_terminal_size()
    assert width > 0  # Should have fallen back to default


def test_safe_read_file(tmp_path: Path) -> None:
    """
    Test safe file reading.

    Args:
        tmp_path: Pytest's temporary directory fixture
    """
    # Create a test file
    test_file = tmp_path / "test_file.txt"
    test_content = "Test content\nWith multiple lines"
    test_file.write_text(test_content)

    # Read file safely
    content = safe_read_file(str(test_file))
    assert content == test_content

    # Test with nonexistent file - should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        safe_read_file(str(tmp_path / "nonexistent.txt"))


def test_print_figlet(capfd: Any) -> None:
    """
    Test print_figlet function.

    Args:
        capfd: Pytest's built-in fixture to capture stdout/stderr
    """
    from figlet_forge import print_figlet

    # Test basic output
    print_figlet("X")
    out, _ = capfd.readouterr()
    assert out.strip()  # Output should not be empty

    # Test with color
    print_figlet("X", colors="RED")
    out, _ = capfd.readouterr()
    assert "\033[" in out  # Should contain ANSI codes


if __name__ == "__main__":
    unittest.main()
