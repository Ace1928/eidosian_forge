"""
Tests for color adjustment capabilities.

Ensures that color detection and adjustment work correctly across
different terminal environments and color depth scenarios.
"""

import os
import re
from unittest.mock import patch

import pytest

from figlet_forge.compat.colour_adjuster import (
    ColourAdjuster,
    adapt_colors_to_terminal,
    strip_colors,
    supports_color,
)


@pytest.fixture(autouse=True)
def _isolated_color_env(monkeypatch):
    """Make color-related tests deterministic across host environments."""
    for key in (
        "NO_COLOR",
        "FORCE_COLOR",
        "COLORTERM",
        "TERM",
        "ANSICON",
        "WT_SESSION",
        "TERM_PROGRAM",
    ):
        monkeypatch.delenv(key, raising=False)


class TestColorDetection:
    """Test color support detection."""

    def test_default_initialization(self):
        """Test default initialization of ColourAdjuster."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("figlet_forge.compat.colour_adjuster.terminal") as mock_terminal:
                mock_terminal.supports_color = True
                mock_terminal.color_depth = 256

                adjuster = ColourAdjuster()
                assert adjuster.supports_color is True
                assert adjuster.effective_color_depth == 256

    @pytest.mark.parametrize(
        "env_vars,expected_mode",
        [
            ({"NO_COLOR": "1"}, "no-color"),
            ({"FORCE_COLOR": "0"}, "no-color"),
            ({"FORCE_COLOR": "1"}, "16"),
            ({"FORCE_COLOR": "2"}, "256"),
            ({"FORCE_COLOR": "3"}, "truecolor"),
            ({"COLORTERM": "truecolor"}, "truecolor"),
            ({"COLORTERM": "24bit"}, "truecolor"),
            ({}, None),  # Default case - no forced mode
        ],
    )
    def test_force_mode_detection(self, env_vars, expected_mode):
        """Test detection of forced color modes."""
        with patch.dict(os.environ, env_vars, clear=True):
            adjuster = ColourAdjuster()
            assert adjuster._get_force_mode() == expected_mode

    @pytest.mark.parametrize(
        "force_mode,terminal_supports,expected",
        [
            ("no-color", True, False),  # NO_COLOR overrides terminal support
            ("16", False, True),  # FORCE_COLOR=1 enables colors
            ("256", False, True),  # FORCE_COLOR=2 enables colors
            ("truecolor", False, True),  # FORCE_COLOR=3 enables colors
            (None, True, True),  # Terminal capability is used
            (None, False, False),  # Terminal capability is used
        ],
    )
    def test_supports_color(self, force_mode, terminal_supports, expected):
        """Test color support detection with various configurations."""
        with patch.dict(os.environ, {}, clear=True):
            adjuster = ColourAdjuster()
            adjuster._force_mode = force_mode
            adjuster._supports_color = terminal_supports

            assert adjuster.supports_color is expected

    @pytest.mark.parametrize(
        "force_mode,terminal_depth,expected",
        [
            ("no-color", 256, 0),  # NO_COLOR forces 0 color depth
            ("16", 0, 16),  # FORCE_COLOR=1 sets 16 colors
            ("256", 0, 256),  # FORCE_COLOR=2 sets 256 colors
            ("truecolor", 0, 16777216),  # FORCE_COLOR=3 sets 16777216 colors
            (None, 16, 16),  # Use terminal's capability
            (None, 256, 256),  # Use terminal's capability
            (None, 16777216, 16777216),  # Use terminal's capability
        ],
    )
    def test_effective_color_depth(self, force_mode, terminal_depth, expected):
        """Test effective color depth calculation."""
        with patch.dict(os.environ, {}, clear=True):
            adjuster = ColourAdjuster()
            adjuster._force_mode = force_mode
            adjuster._supports_color = terminal_depth > 0
            adjuster._color_depth = terminal_depth

            assert adjuster.effective_color_depth == expected


class TestColorStripping:
    """Test stripping of ANSI color codes."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("\033[31mRed\033[0m", "Red"),  # Basic red color
            ("\033[1;32mBold green\033[0m", "Bold green"),  # Bold green
            ("\033[38;5;200mPink\033[0m", "Pink"),  # 256-color pink
            ("\033[38;2;255;100;0mOrange\033[0m", "Orange"),  # 24-bit orange
            ("No colors here", "No colors here"),  # No colors to strip
            (
                "\033[31mRed\033[0m \033[32mGreen\033[0m \033[34mBlue\033[0m",
                "Red Green Blue",
            ),  # Multiple colors
        ],
    )
    def test_strip_colors(self, text, expected):
        """Test stripping ANSI color codes."""
        adjuster = ColourAdjuster()
        assert adjuster.strip_colors(text) == expected


class TestColorDowngrading:
    """Test downgrading colors to match terminal capabilities."""

    def test_no_color_support(self):
        """Test behavior when terminal doesn't support colors."""
        adjuster = ColourAdjuster()
        adjuster._supports_color = False

        text = "\033[31mRed\033[0m \033[32mGreen\033[0m"
        expected = "Red Green"  # All colors should be stripped
        assert adjuster.downgrade_colors(text) == expected

    def test_truecolor_support(self):
        """Test behavior with truecolor support."""
        adjuster = ColourAdjuster()
        adjuster._supports_color = True
        adjuster._color_depth = 16777216  # 24-bit/truecolor

        # Text with 24-bit colors should remain unchanged
        text = "\033[38;2;255;100;0mOrange\033[0m"
        assert adjuster.downgrade_colors(text) == text

    def test_downgrade_24bit_to_256(self):
        """Test downgrading 24-bit colors to 256 colors."""
        adjuster = ColourAdjuster()
        adjuster._supports_color = True
        adjuster._color_depth = 256

        with patch.object(adjuster, "_rgb_to_256", return_value=173):  # Example value
            text = "\033[38;2;255;100;0mOrange\033[0m"
            expected = text.replace("38;2;255;100;0", "38;5;173")
            assert adjuster.downgrade_colors(text) == expected

    def test_downgrade_256_to_16(self):
        """Test downgrading 256 colors to 16 colors."""
        adjuster = ColourAdjuster()
        adjuster._supports_color = True
        adjuster._color_depth = 16

        with patch.object(adjuster, "_256_to_16", return_value="31"):  # Red
            text = "\033[38;5;173mOrange\033[0m"
            expected = text.replace("38;5;173", "31")
            assert adjuster.downgrade_colors(text) == expected


class TestColorConversion:
    """Test color conversion functions."""

    @pytest.mark.parametrize(
        "r,g,b,expected_range",
        [
            (0, 0, 0, (16, 16)),  # Black should map to around 16
            (255, 255, 255, (231, 231)),  # White should map to around 231
            (255, 0, 0, (196, 196)),  # Pure red
            (0, 255, 0, (46, 46)),  # Pure green
            (0, 0, 255, (21, 21)),  # Pure blue
        ],
    )
    def test_rgb_to_256(self, r, g, b, expected_range):
        """Test conversion from RGB to 256-color code."""
        adjuster = ColourAdjuster()
        result = adjuster._rgb_to_256(r, g, b)

        # Check result is within expected range (allowing some flexibility)
        min_val, max_val = expected_range
        assert (
            min_val <= result <= max_val
        ), f"Expected {result} to be between {min_val} and {max_val}"

    @pytest.mark.parametrize(
        "color,base_code,expected",
        [
            (0, "3", "30"),  # Black foreground
            (1, "3", "31"),  # Red foreground
            (9, "3", "31;1"),  # Bright red foreground (8+1, bright bit)
            (16, "3", "30"),  # First color in 256-color cube maps to black in current converter
            (232, "3", "30"),  # First grayscale entry maps to black in current converter
            (0, "4", "40"),  # Black background
            (9, "4", "41;1"),  # Bright red background
        ],
    )
    def test_256_to_16(self, color, base_code, expected):
        """Test conversion from 256-color code to 16-color ANSI code."""
        adjuster = ColourAdjuster()
        result = adjuster._256_to_16(color, base_code)
        assert result == expected

    @pytest.mark.parametrize(
        "r,g,b,base_code,expected_pattern",
        [
            (255, 0, 0, "3", r"3(1|1;1)"),  # Red (bright or regular)
            (0, 255, 0, "3", r"3(2|2;1)"),  # Green (bright or regular)
            (0, 0, 255, "3", r"3(4|4;1)"),  # Blue (bright or regular)
            (255, 255, 0, "3", r"3(3|3;1)"),  # Yellow (bright or regular)
            (0, 0, 0, "4", r"4(0|0;)"),  # Black background
        ],
    )
    def test_rgb_to_16(self, r, g, b, base_code, expected_pattern):
        """Test conversion from RGB to 16-color ANSI code."""
        adjuster = ColourAdjuster()
        result = adjuster._rgb_to_16(r, g, b, base_code)
        assert re.match(
            expected_pattern, result
        ), f"Expected {result} to match {expected_pattern}"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_supports_color(self):
        """Test the supports_color convenience function."""
        with patch("figlet_forge.compat.colour_adjuster.colour") as mock_colour:
            mock_colour.supports_color = True
            assert supports_color() is True

            mock_colour.supports_color = False
            assert supports_color() is False

    def test_strip_colors(self):
        """Test the strip_colors convenience function."""
        with patch("figlet_forge.compat.colour_adjuster.colour") as mock_colour:
            mock_colour.strip_colors.return_value = "Stripped"
            assert strip_colors("Test") == "Stripped"
            mock_colour.strip_colors.assert_called_with("Test")

    def test_adapt_colors_to_terminal(self):
        """Test the adapt_colors_to_terminal convenience function."""
        with patch("figlet_forge.compat.colour_adjuster.colour") as mock_colour:
            mock_colour.downgrade_colors.return_value = "Adapted"
            assert adapt_colors_to_terminal("Test") == "Adapted"
            mock_colour.downgrade_colors.assert_called_with("Test")
