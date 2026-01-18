"""
Unit tests for color effects in Figlet Forge.

These tests verify the color effects functionality including gradient,
rainbow, pulse effects and pattern highlighting.
"""

import unittest
from unittest.mock import patch

import pytest

from figlet_forge.color.effects import (
    _parse_color_to_rgb,
    color_style_apply,
    gradient_colorize,
    highlight_pattern,
    pulse_colorize,
    rainbow_colorize,
)
from figlet_forge.core.exceptions import InvalidColor


class TestColorEffects(unittest.TestCase):
    """Test color effects functionality."""

    def test_rainbow_colorize(self):
        """Test rainbow coloring effect."""
        # Test with simple input
        result = rainbow_colorize("ABC")
        self.assertTrue("\033[" in result)
        self.assertNotEqual(result, "ABC")

        # Test with multiline input
        result = rainbow_colorize("A\nB\nC")
        self.assertTrue("\033[" in result)
        self.assertTrue("\n" in result)

        # Test with empty input
        result = rainbow_colorize("")
        self.assertEqual(result, "")

    def test_gradient_colorize(self):
        """Test gradient coloring effect."""
        # Test with named colors
        result = gradient_colorize("ABC", "RED", "BLUE")
        self.assertTrue("\033[" in result)
        self.assertNotEqual(result, "ABC")

        # Test with RGB values
        result = gradient_colorize("ABC", "255;0;0", "0;0;255")
        self.assertTrue("\033[" in result)
        self.assertNotEqual(result, "ABC")

        # Test with invalid colors
        with self.assertRaises(InvalidColor):
            gradient_colorize("ABC", "INVALID", "BLUE")

        # Test with empty input
        result = gradient_colorize("", "RED", "BLUE")
        self.assertEqual(result, "")

    def test_pulse_colorize(self):
        """Test pulse coloring effect."""
        # Test with named color
        result = pulse_colorize("ABCDE", "RED")
        self.assertTrue("\033[" in result)
        self.assertNotEqual(result, "ABCDE")

        # Test with different intensity levels
        result1 = pulse_colorize("ABCDE", "BLUE", 3)
        result2 = pulse_colorize("ABCDE", "BLUE", 5)
        self.assertNotEqual(result1, result2)

        # Test with invalid color
        with self.assertRaises(InvalidColor):
            pulse_colorize("ABC", "INVALID")

        # Test with empty input
        result = pulse_colorize("", "RED")
        self.assertEqual(result, "")

    def test_highlight_pattern(self):
        """Test pattern highlighting."""
        # Test with simple pattern
        text = "Hello world"
        result = highlight_pattern(text, "world", "RED")
        self.assertTrue("\033[" in result)
        self.assertNotEqual(result, text)

        # Test with regex pattern
        text = "Hello 123 world 456"
        result = highlight_pattern(text, r"\d+", "BLUE")
        self.assertTrue("\033[" in result)
        self.assertNotEqual(result, text)

        # Test case sensitivity
        text = "Hello World"
        result1 = highlight_pattern(text, "world", "RED")  # Case-sensitive (no match)
        result2 = highlight_pattern(
            text, "world", "RED", case_sensitive=False
        )  # Not case-sensitive
        self.assertEqual(result1, text)  # No highlighting
        self.assertNotEqual(result2, text)  # Has highlighting

        # Test with invalid color
        with self.assertRaises(InvalidColor):
            highlight_pattern(text, "Hello", "INVALID")

        # Test with empty input
        result = highlight_pattern("", "pattern", "RED")
        self.assertEqual(result, "")

    def test_parse_color_to_rgb(self):
        """Test color parsing to RGB."""
        # Test named colors
        rgb = _parse_color_to_rgb("RED")
        self.assertEqual(rgb, (255, 0, 0))

        # Test RGB format
        rgb = _parse_color_to_rgb("100;150;200")
        self.assertEqual(rgb, (100, 150, 200))

        # Test invalid format
        rgb = _parse_color_to_rgb("INVALID")
        self.assertIsNone(rgb)

    def test_color_style_apply(self):
        """Test applying predefined color styles."""
        # Test with rainbow style
        with patch(
            "figlet_forge.color.effects.rainbow_colorize", return_value="RAINBOW"
        ):
            result = color_style_apply("ABC", "rainbow")
            self.assertEqual(result, "RAINBOW")

        # Test with gradient style
        with patch(
            "figlet_forge.color.effects.gradient_colorize", return_value="GRADIENT"
        ):
            result = color_style_apply("ABC", "red_to_blue")
            self.assertEqual(result, "GRADIENT")

        # Test with fg/bg style
        with patch("figlet_forge.color.effects._apply_fg_bg", return_value="FGBG"):
            result = color_style_apply("ABC", "red_on_black")
            self.assertEqual(result, "FGBG")

        # Test with invalid style
        with self.assertRaises(ValueError):
            color_style_apply("ABC", "invalid_style")


@pytest.mark.parametrize(
    "input_text,color,expected_contains",
    [
        ("Hello", "RED", "\033[31m"),  # Simple text with red color
        ("AB\nCD", "GREEN", "\033[32m"),  # Multiline with green color
        ("12345", "BLUE", "\033[34m"),  # Numbers with blue color
    ],
)
def test_rainbow_colorize_parametrized(input_text, color, expected_contains):
    """
    Test rainbow_colorize with different inputs using parametrization.

    Args:
        input_text: Input text to colorize
        color: Not used directly but ensures test variations
        expected_contains: ANSI code expected in the output
    """
    # For rainbow, we're just checking that colorization happens
    result = rainbow_colorize(input_text)

    # Rainbow should contain ANSI codes
    assert "\033[" in result

    # The result should be different from the input
    assert result != input_text


@pytest.mark.parametrize(
    "start_color,end_color,raises_error",
    [
        ("RED", "BLUE", False),  # Valid named colors
        ("255;0;0", "0;0;255", False),  # Valid RGB values
        ("RED", "0;0;255", False),  # Mixed formats
        ("INVALID", "BLUE", True),  # Invalid start color
        ("RED", "INVALID", True),  # Invalid end color
    ],
)
def test_gradient_colorize_error_handling(start_color, end_color, raises_error):
    """
    Test error handling in gradient_colorize.

    Args:
        start_color: Starting color
        end_color: Ending color
        raises_error: Whether an error should be raised
    """
    text = "Test"

    if raises_error:
        with pytest.raises(InvalidColor):
            gradient_colorize(text, start_color, end_color)
    else:
        try:
            result = gradient_colorize(text, start_color, end_color)
            assert "\033[" in result
            assert result != text
        except InvalidColor:
            pytest.fail("Unexpected InvalidColor exception")


if __name__ == "__main__":
    unittest.main()
