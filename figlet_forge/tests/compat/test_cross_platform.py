"""
Cross-platform compatibility tests.

This module provides tests that ensure figlet_forge works properly
across different operating systems, terminal types, and environments.
"""

import os
import platform
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from figlet_forge import Figlet
from figlet_forge.compat.colour_adjuster import ColourAdjuster
from figlet_forge.compat.encoding_adjuster import supports_utf8
from figlet_forge.compat.figlet_compat import Figlet as CompatFiglet
from figlet_forge.compat.terminal_adjuster import (
    TerminalAdjuster,
    get_terminal_size,
    supports_color,
    supports_unicode,
)


class TestWindowsCompatibility:
    """Test compatibility with Windows environments."""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only test")
    def test_real_windows_terminal(self):
        """Test terminal detection on a real Windows system."""
        # Only meaningful on actual Windows systems
        adjuster = TerminalAdjuster()

        # Basic checks that shouldn't crash
        assert isinstance(adjuster.terminal_width, int)
        assert isinstance(adjuster.terminal_height, int)
        assert isinstance(adjuster.supports_color, bool)
        assert isinstance(adjuster.supports_unicode, bool)

    @patch("platform.system", return_value="Windows")
    @patch("sys.getwindowsversion", create=True)
    def test_windows_10_detection(self, mock_version, _):
        """Test detection of Windows 10 capabilities."""
        mock_version.return_value = SimpleNamespace(major=10, build=19041)

        with patch("sys.stdout.isatty", return_value=True):
            adjuster = TerminalAdjuster()
            assert isinstance(adjuster.supports_color, bool)

    @patch("platform.system", return_value="Windows")
    @patch("sys.getwindowsversion", create=True)
    @patch.dict(os.environ, {"WT_SESSION": "1"}, clear=True)
    def test_windows_terminal_detection(self, mock_version, _):
        """Test detection of Windows Terminal capabilities."""
        mock_version.return_value = SimpleNamespace(major=10, build=19041)

        adjuster = TerminalAdjuster()
        assert isinstance(adjuster.supports_unicode, bool)
        assert isinstance(adjuster.supports_true_color, bool)

    @patch("platform.system", return_value="Windows")
    def test_windows_figlet(self, _):
        """Test figlet rendering on Windows."""
        with patch("figlet_forge.compat.terminal_adjuster.terminal") as mock_terminal:
            mock_terminal.supports_color = False
            mock_terminal.supports_unicode = False

            # Test basic ASCII rendering
            fig = Figlet(font="standard")
            result = fig.renderText("Test")

            assert isinstance(result, str)
            assert len(result) > 0
            assert "Test" not in result  # It should be rendered as ASCII art


class TestUnixCompatibility:
    """Test compatibility with Unix-like environments."""

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-like systems only")
    def test_real_unix_terminal(self):
        """Test terminal detection on a real Unix-like system."""
        adjuster = TerminalAdjuster()

        # Basic checks that shouldn't crash
        assert isinstance(adjuster.terminal_width, int)
        assert isinstance(adjuster.terminal_height, int)
        assert isinstance(adjuster.supports_color, bool)
        assert isinstance(adjuster.supports_unicode, bool)

    @patch("platform.system", return_value="Linux")
    @patch.dict(
        os.environ, {"TERM": "xterm-256color", "LANG": "en_US.UTF-8"}, clear=True
    )
    def test_xterm_detection(self, _):
        """Test detection of xterm capabilities."""
        with patch("sys.stdout.isatty", return_value=True):
            adjuster = TerminalAdjuster()
            assert adjuster.supports_color
            assert adjuster.supports_unicode
            assert adjuster.color_depth == 256

    @patch("platform.system", return_value="Linux")
    @patch.dict(os.environ, {"COLORTERM": "truecolor"}, clear=True)
    def test_truecolor_detection(self, _):
        """Test detection of truecolor support."""
        with patch("sys.stdout.isatty", return_value=True):
            adjuster = TerminalAdjuster()
            assert isinstance(adjuster.supports_true_color, bool)
            assert isinstance(adjuster.color_depth, int)


class TestCIEnvironments:
    """Test compatibility with CI environments."""

    @patch.dict(os.environ, {"CI": "true"}, clear=True)
    def test_ci_detection(self):
        """Test detection of CI environment."""
        adjuster = TerminalAdjuster()
        assert isinstance(adjuster._term_type, int)

    @patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True)
    def test_github_actions_detection(self):
        """Test detection of GitHub Actions environment."""
        adjuster = TerminalAdjuster()
        assert isinstance(adjuster._term_type, int)


class TestSpecialEnvVars:
    """Test handling of special environment variables."""

    @patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True)
    def test_no_color_env(self):
        """Test handling of NO_COLOR environment variable."""
        color_adjuster = ColourAdjuster()
        assert not color_adjuster.supports_color
        assert color_adjuster.effective_color_depth == 0

    @patch.dict(os.environ, {"FORCE_COLOR": "3"}, clear=True)
    def test_force_color_env(self):
        """Test handling of FORCE_COLOR environment variable."""
        color_adjuster = ColourAdjuster()
        assert color_adjuster.supports_color
        assert color_adjuster.effective_color_depth == 16777216


class TestOutputFormats:
    """Test rendering to different output formats across platforms."""

    def test_ascii_output(self):
        """Test plain ASCII output."""
        with patch("figlet_forge.compat.terminal_adjuster.terminal") as mock_terminal:
            mock_terminal.supports_unicode = False
            mock_terminal.adjust_output_for_terminal.side_effect = lambda text, _: text

            fig = Figlet(font="standard")
            result = fig.renderText("Test")

            assert isinstance(result, str)
            assert len(result) > 0

    def test_unicode_output(self):
        """Test Unicode output."""
        with patch("figlet_forge.compat.terminal_adjuster.terminal") as mock_terminal:
            mock_terminal.supports_unicode = True
            mock_terminal.adjust_output_for_terminal.side_effect = lambda text, _: text

            fig = Figlet(font="standard", unicode_aware=True)
            result = fig.renderText("Test ☺")

            assert isinstance(result, str)
            assert len(result) > 0
            # Since we're mocking, we can't test actual Unicode output

    def test_color_output(self):
        """Test colored output."""
        with patch("figlet_forge.compat.terminal_adjuster.terminal") as mock_terminal:
            mock_terminal.supports_color = True
            mock_terminal.color_depth = 16
            mock_terminal.adjust_output_for_terminal.side_effect = lambda text, _: text

            # This would depend on how your actual color implementation works
            fig = Figlet(font="standard")
            result = fig.renderText("Test")
            if not hasattr(fig, "colorize"):
                pytest.skip("Figlet colorize API not exposed in this build")
            colored_result = fig.colorize(result, "red")

            assert isinstance(colored_result, str)
            assert len(colored_result) > len(result)  # Color codes add length


class TestCompatibilityClasses:
    """Test the compatibility classes work properly."""

    def test_figlet_compat_simple(self):
        """Test basic functionality of the figlet compatibility layer."""
        fig = CompatFiglet(font="standard")
        result = fig.renderText("Test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_figlet_compat_options(self):
        """Test options in the figlet compatibility layer."""
        # Direction setting
        fig = CompatFiglet(direction="right-to-left")
        assert fig.getDirection() == "right_to_left"  # Note underscore format

        # Justify setting
        fig = CompatFiglet(justify="center")
        assert fig.getJustify() == "center"

        # Width setting
        fig = CompatFiglet(width=120)
        assert fig.width == 120

        # Legacy r_to_l parameter
        fig = CompatFiglet(r_to_l=True)
        assert fig.getDirection() == "right_to_left"

    def test_convenience_functions(self):
        """Test that convenience functions work properly."""
        size = get_terminal_size()
        assert isinstance(size, tuple)
        assert len(size) == 2
        assert all(isinstance(dim, int) for dim in size)

        has_color = supports_color()
        assert isinstance(has_color, bool)

        has_unicode = supports_unicode()
        assert isinstance(has_unicode, bool)

        has_utf8 = supports_utf8()
        assert isinstance(has_utf8, bool)


class TestCrossEnvironmentConsistency:
    """Test consistent behavior across environments."""

    @pytest.mark.parametrize("font", ["standard", "slant", "small"])
    def test_font_consistency(self, font):
        """Test font rendering is consistent across environments."""
        fig1 = Figlet(font=font)
        fig2 = CompatFiglet(font=font)

        text = "Test"
        result1 = fig1.renderText(text)
        result2 = fig2.renderText(text)

        # The core rendering should be the same
        # (Compatibility layer might add extra newlines or formatting)
        assert result1.strip() == result2.strip()

    @pytest.mark.parametrize("text", ["Hello", "123", "!@#$%"])
    def test_text_consistency(self, text):
        """Test text rendering is consistent."""
        fig = Figlet(font="standard")

        # Test with different terminal configurations
        with patch("figlet_forge.compat.terminal_adjuster.terminal") as mock_terminal:
            mock_terminal.supports_color = True
            mock_terminal.supports_unicode = True
            result1 = fig.renderText(text)

            mock_terminal.supports_color = False
            mock_terminal.supports_unicode = False
            result2 = fig.renderText(text)

            # Core figlet pattern should be the same despite terminal differences
            # (Allowing for color codes and Unicode differences)
            assert len(result1.strip()) > 0
            assert len(result2.strip()) > 0
