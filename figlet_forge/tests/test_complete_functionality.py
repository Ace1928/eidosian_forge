"""
Integration tests for complete Figlet Forge functionality.

These tests verify that all components of Figlet Forge work together correctly,
ensuring full functionality and compatibility with all features.
"""

import sys
import unittest
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest

# Add project root to path for import resolution
sys.path.insert(0, str(Path(__file__).parent.parent))

from figlet_forge import Figlet
from figlet_forge.cli.main import main
from figlet_forge.color.effects import (
    color_style_apply,
    gradient_colorize,
    highlight_pattern,
    pulse_colorize,
    rainbow_colorize,
)
from figlet_forge.compat import figlet_format
from figlet_forge.core.figlet_string import FigletString


class TestCompleteFunctionality(unittest.TestCase):
    """Test complete functionality of Figlet Forge."""

    def test_basic_figlet_rendering(self):
        """Test that basic figlet rendering works."""
        fig = Figlet(font="standard")
        result = fig.renderText("Test")

        self.assertIsInstance(result, FigletString)
        self.assertTrue(result)  # Not empty

    def test_color_effects(self):
        """Test that color effects work correctly."""
        text = "Test"
        fig = Figlet(font="standard")
        result = fig.renderText(text)

        # Test rainbow effect
        rainbow = rainbow_colorize(str(result))
        self.assertIn("\033[", rainbow)  # Contains ANSI codes

        # Test gradient effect
        gradient = gradient_colorize(str(result), "RED", "BLUE")
        self.assertIn("\033[", gradient)  # Contains ANSI codes

        # Test pulse effect
        pulse = pulse_colorize(str(result), "RED")
        self.assertIn("\033[", pulse)  # Contains ANSI codes

        # Test highlighting
        highlighted = highlight_pattern(str(result), "Test", "GREEN")
        self.assertIn("\033[", highlighted)  # Contains ANSI codes

    def test_transformations(self):
        """Test that transformations work correctly."""
        fig = Figlet(font="standard")
        result = fig.renderText("Test")

        # Test reverse
        reversed_text = result.reverse()
        self.assertIsInstance(reversed_text, FigletString)
        self.assertNotEqual(str(reversed_text), str(result))

        # Test flip
        flipped_text = result.flip()
        self.assertIsInstance(flipped_text, FigletString)
        self.assertNotEqual(str(flipped_text), str(result))

        # Test border
        bordered_text = result.border()
        self.assertIsInstance(bordered_text, FigletString)
        self.assertIn("┌", str(bordered_text))

        # Test shadow
        shadowed_text = result.shadow()
        self.assertIsInstance(shadowed_text, FigletString)
        self.assertNotEqual(str(shadowed_text), str(result))

    def test_cli_functionality(self):
        """Test that CLI functionality works."""
        # Test basic CLI rendering
        with patch("sys.stdout") as mock_stdout:
            exit_code = main(["Test"])
            self.assertEqual(exit_code, 0)
            mock_stdout.write.assert_called()

        # Test with color option
        with patch("sys.stdout") as mock_stdout:
            exit_code = main(["--color=RED", "Test"])
            self.assertEqual(exit_code, 0)
            mock_stdout.write.assert_called()

        # Test with transformation options
        with patch("sys.stdout") as mock_stdout:
            exit_code = main(["--reverse", "--flip", "--border=single", "Test"])
            self.assertEqual(exit_code, 0)
            mock_stdout.write.assert_called()

    def test_pyfiglet_compatibility(self):
        """Test compatibility with pyfiglet."""
        # Test figlet_format function
        result = figlet_format("Test")
        self.assertTrue(result)  # Not empty
        self.assertIsInstance(result, str)


@pytest.fixture
def run_cli() -> callable:
    """
    Fixture to run CLI commands with captured output.

    Returns:
        Function to run CLI commands
    """

    def _run(args: List[str]) -> Tuple[int, str, str]:
        """Run CLI with args and return exit code, stdout, and stderr."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                exit_code = main(args)
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 1

        return exit_code, stdout.getvalue(), stderr.getvalue()

    return _run


@pytest.mark.parametrize(
    "cli_args",
    [
        ["Test"],  # Basic rendering
        ["--font=standard", "Test"],  # Font selection
        ["--color=RED", "Test"],  # Color
        ["--color=rainbow", "Test"],  # Rainbow effect
        ["--reverse", "Test"],  # Reverse transformation
        ["--flip", "Test"],  # Flip transformation
        ["--border=single", "Test"],  # Border
        ["--shade", "Test"],  # Shadow
        ["--unicode", "Test ☺"],  # Unicode support
        ["--list-fonts"],  # List fonts
        ["--version"],  # Version information
    ],
)
def test_cli_commands(run_cli, cli_args):
    """
    Test various CLI commands.

    Args:
        run_cli: Fixture to run CLI commands
        cli_args: CLI arguments to test
    """
    exit_code, stdout, stderr = run_cli(cli_args)

    # All commands should succeed
    assert exit_code == 0, f"Command failed: {' '.join(cli_args)}\nStderr: {stderr}"

    # Standard output should have content
    assert stdout.strip(), f"No output for command: {' '.join(cli_args)}"


@pytest.mark.parametrize(
    "font_name",
    [
        "standard",  # Always should exist
        "slant",
        "small",
        "mini",
        "big",
        "banner",
    ],
)
def test_standard_fonts(font_name):
    """
    Test that standard fonts work correctly.

    Args:
        font_name: Font to test
    """
    try:
        fig = Figlet(font=font_name)
        result = fig.renderText("Test")

        # Font should render correctly
        assert result, f"Empty result from font {font_name}"
        assert isinstance(result, FigletString)
    except Exception as e:
        pytest.fail(f"Font {font_name} failed: {e}")


@pytest.mark.parametrize(
    "color_style",
    [
        "rainbow",
        "red_to_blue",
        "yellow_to_green",
        "magenta_to_cyan",
        "white_to_blue",
        "red_on_black",
        "green_on_black",
        "yellow_on_blue",
    ],
)
def test_color_styles(color_style):
    """
    Test that color styles work correctly.

    Args:
        color_style: Color style to test
    """
    fig = Figlet(font="standard")
    result = fig.renderText("Test")

    try:
        if color_style == "rainbow":
            colored = rainbow_colorize(str(result))
        elif "_to_" in color_style:
            start, end = color_style.split("_to_")
            colored = gradient_colorize(str(result), start.upper(), end.upper())
        else:
            colored = color_style_apply(str(result), color_style)

        # Should contain ANSI color codes
        assert "\033[" in colored, f"No color codes in {color_style} output"
    except Exception as e:
        pytest.fail(f"Color style {color_style} failed: {e}")


def test_showcase_feature(run_cli):
    """Test the showcase feature."""
    # This would be slow, so we'll mock it
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        # Fix: Use --sample-color=RED instead of separate arguments
        exit_code, stdout, stderr = run_cli(
            ["--showcase", "--sample-text=Test", "--sample-color=RED"]
        )

        # Even if there are warnings, it should still succeed with code 0
        assert exit_code == 0, f"Failed with stderr: {stderr}"
        mock_showcase.assert_called_once()


if __name__ == "__main__":
    unittest.main()
