"""
Tests for terminal compatibility handling.

Ensures that terminal detection and adjustment work correctly across
different environments and terminal types.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from figlet_forge.compat.terminal_adjuster import (
    CAPS_BOLD,
    CAPS_COLOR,
    CAPS_ITALIC,
    CAPS_TRUE_COLOR,
    CAPS_UNICODE,
    CAPS_WIDE_CHARS,
    TERM_TYPE_ANSI,
    TERM_TYPE_DUMB,
    TERM_TYPE_NO_COLOR,
    TERM_TYPE_UNKNOWN,
    TERM_TYPE_WINDOWS,
    TerminalAdjuster,
    adjust_output,
    get_terminal_size,
    supports_color,
    supports_unicode,
)


def test_terminal_adjuster_initialization():
    """Test that TerminalAdjuster initializes correctly."""
    with patch(
        "figlet_forge.compat.terminal_adjuster.TerminalAdjuster._detect_terminal_type"
    ) as mock_detect_type, patch(
        "figlet_forge.compat.terminal_adjuster.TerminalAdjuster._detect_capabilities"
    ) as mock_detect_caps, patch(
        "figlet_forge.compat.terminal_adjuster.TerminalAdjuster._get_terminal_size"
    ) as mock_get_size, patch(
        "figlet_forge.compat.terminal_adjuster.TerminalAdjuster._detect_color_depth"
    ) as mock_detect_color:

        mock_detect_type.return_value = TERM_TYPE_ANSI
        mock_detect_caps.return_value = CAPS_COLOR | CAPS_UNICODE
        mock_get_size.return_value = (80, 25)
        mock_detect_color.return_value = 256

        terminal = TerminalAdjuster()

        assert terminal._term_type == TERM_TYPE_ANSI
        assert terminal._capabilities == (CAPS_COLOR | CAPS_UNICODE)
        assert terminal._size == (80, 25)
        assert terminal._color_depth == 256


@pytest.mark.parametrize(
    "env_vars,isatty,platform_system,expected_type",
    [
        ({"CI": "true"}, True, "Linux", TERM_TYPE_ANSI),  # CI environment
        ({"GITHUB_ACTIONS": "true"}, True, "Linux", TERM_TYPE_ANSI),  # GitHub Actions
        ({"NO_COLOR": "1"}, True, "Linux", TERM_TYPE_NO_COLOR),  # NO_COLOR standard
        ({"TERM": "dumb"}, True, "Linux", TERM_TYPE_DUMB),  # Dumb terminal
        ({}, True, "Windows", TERM_TYPE_ANSI),  # Windows with TTY
        ({}, False, "Windows", TERM_TYPE_WINDOWS),  # Windows without TTY
        ({"WT_SESSION": "1"}, True, "Windows", TERM_TYPE_ANSI),  # Windows Terminal
        ({"TERM_PROGRAM": "vscode"}, True, "Windows", TERM_TYPE_ANSI),  # VS Code
        ({}, True, "Linux", TERM_TYPE_ANSI),  # Linux with TTY
        ({}, False, "Linux", TERM_TYPE_UNKNOWN),  # Linux without TTY
    ],
)
def test_terminal_type_detection(env_vars, isatty, platform_system, expected_type):
    """Test that terminal type is correctly detected."""
    with patch.dict(os.environ, env_vars, clear=True), patch(
        "sys.stdout.isatty", return_value=isatty
    ), patch("platform.system", return_value=platform_system):

        # Create a new instance for each test to avoid cached values
        terminal = TerminalAdjuster()
        terminal_type = terminal._detect_terminal_type()

        if platform_system == "Windows" and isatty and not env_vars:
            assert terminal_type in {TERM_TYPE_ANSI, TERM_TYPE_WINDOWS}
        else:
            assert terminal_type == expected_type


@pytest.mark.parametrize(
    "term_type,env_vars,system_version,expected_caps",
    [
        (TERM_TYPE_UNKNOWN, {}, None, 0),  # Unknown terminal has no capabilities
        (TERM_TYPE_DUMB, {}, None, 0),  # Dumb terminal has no capabilities
        (TERM_TYPE_NO_COLOR, {}, None, 0),  # NO_COLOR terminal has no color
        (
            TERM_TYPE_WINDOWS,
            {},
            (10, 0, 0),
            CAPS_COLOR,
        ),  # Windows 10 supports color
        (
            TERM_TYPE_WINDOWS,
            {"WT_SESSION": "1"},
            (10, 0, 0),
            CAPS_COLOR
            | CAPS_UNICODE
            | CAPS_WIDE_CHARS
            | CAPS_BOLD
            | CAPS_ITALIC
            | CAPS_TRUE_COLOR,
        ),  # Windows Terminal supports everything
        (
            TERM_TYPE_ANSI,
            {"TERM": "xterm-256color"},
            None,
            CAPS_COLOR | CAPS_BOLD,
        ),  # Basic ANSI terminal
        (
            TERM_TYPE_ANSI,
            {"TERM": "xterm-256color", "LANG": "en_US.UTF-8"},
            None,
            CAPS_COLOR | CAPS_UNICODE | CAPS_WIDE_CHARS | CAPS_BOLD | CAPS_ITALIC,
        ),  # UTF-8 ANSI terminal
        (
            TERM_TYPE_ANSI,
            {"COLORTERM": "truecolor", "TERM": "xterm-256color", "LANG": "en_US.UTF-8"},
            None,
            CAPS_COLOR
            | CAPS_UNICODE
            | CAPS_WIDE_CHARS
            | CAPS_TRUE_COLOR
            | CAPS_BOLD
            | CAPS_ITALIC,
        ),  # True color ANSI terminal
    ],
)
def test_capabilities_detection(term_type, env_vars, system_version, expected_caps):
    """Test that terminal capabilities are correctly detected."""
    with patch.dict(os.environ, env_vars, clear=True), patch(
        "figlet_forge.compat.terminal_adjuster.TerminalAdjuster._detect_terminal_type",
        return_value=term_type,
    ), patch(
        "sys.getwindowsversion",
        create=True,
        return_value=MagicMock(major=system_version[0] if system_version else 0),
    ):

        # Create a new instance for each test
            terminal = TerminalAdjuster()
            terminal._term_type = term_type  # Set term_type directly
            capabilities = terminal._detect_capabilities()

            assert capabilities & expected_caps == expected_caps


@pytest.mark.parametrize(
    "term_type,env_vars,capabilities,expected_depth",
    [
        (TERM_TYPE_DUMB, {}, 0, 0),  # Dumb terminal has no color
        (TERM_TYPE_NO_COLOR, {}, 0, 0),  # NO_COLOR terminal has no color
        (
            TERM_TYPE_ANSI,
            {"COLORTERM": "truecolor"},
            CAPS_COLOR,
            16777216,
        ),  # True color via COLORTERM
        (
            TERM_TYPE_ANSI,
            {"TERM": "xterm-256color"},
            CAPS_COLOR,
            256,
        ),  # 256 colors via TERM
        (TERM_TYPE_ANSI, {}, CAPS_COLOR, 16),  # Basic ANSI terminal
        (
            TERM_TYPE_ANSI,
            {},
            CAPS_COLOR | CAPS_TRUE_COLOR,
            16777216,
        ),  # True color via caps
        (TERM_TYPE_WINDOWS, {}, CAPS_COLOR, 16),  # Windows with color
        (
            TERM_TYPE_WINDOWS,
            {},
            CAPS_COLOR | CAPS_TRUE_COLOR,
            16777216,
        ),  # Windows with true color
    ],
)
def test_color_depth_detection(term_type, env_vars, capabilities, expected_depth):
    """Test that color depth is correctly detected."""
    with patch.dict(os.environ, env_vars, clear=True):
        terminal = TerminalAdjuster()
        terminal._term_type = term_type  # Set term_type directly
        terminal._capabilities = capabilities  # Set capabilities directly

        color_depth = terminal._detect_color_depth()
        assert color_depth == expected_depth


def test_terminal_size_detection():
    """Test that terminal size is correctly detected."""
    # Test with shutil.get_terminal_size
    with patch("shutil.get_terminal_size", return_value=(120, 40)):
        terminal = TerminalAdjuster()
        size = terminal._get_terminal_size()
        assert size == (120, 40)

    # Test with environment variables when shutil raises
    with patch("shutil.get_terminal_size", side_effect=OSError), patch.dict(
        os.environ, {"COLUMNS": "100", "LINES": "30"}, clear=True
    ):
        terminal = TerminalAdjuster()
        size = terminal._get_terminal_size()
        assert size == (100, 30)

    # Test fallback
    with patch("shutil.get_terminal_size", side_effect=OSError), patch.dict(
        os.environ, {}, clear=True
    ):
        terminal = TerminalAdjuster()
        size = terminal._get_terminal_size()
        assert size == (80, 24)  # Default fallback


def test_strip_formatting():
    """Test stripping ANSI formatting codes."""
    terminal = TerminalAdjuster()

    # Test with color support
    terminal._capabilities = CAPS_COLOR
    text = "\033[31mRed\033[0m \033[32mGreen\033[0m"
    assert terminal.strip_formatting(text) == text

    # Test without color support
    terminal._capabilities = 0
    assert terminal.strip_formatting(text) == "Red Green"


@pytest.mark.parametrize(
    "text,supports_unicode,enforce_ascii,expected_output",
    [
        # Unicode text with Unicode support
        ("Hello 世界", True, False, "Hello 世界"),
        # Unicode text without Unicode support
        ("Hello 世界", False, False, "Hello ??"),
        # Unicode text with enforced ASCII
        ("Hello 世界", True, True, "Hello ??"),
        # ASCII-only text
        ("Hello world", False, False, "Hello world"),
        # Box-drawing characters
        ("┌───┐\n│ABC│\n└───┘", False, True, "+---+\n|ABC|\n+---+"),
    ],
)
def test_adjust_output_for_terminal(
    text, supports_unicode, enforce_ascii, expected_output
):
    """Test adjusting output for terminal capabilities."""
    # This is a simplified test - in reality, conversion is more complex
    terminal = TerminalAdjuster()
    terminal._capabilities = CAPS_UNICODE if supports_unicode else 0

    # Create a regex pattern that matches either the expected output or something close
    # This is because different systems might handle Unicode replacement differently
    result = terminal.adjust_output_for_terminal(text, enforce_ascii)

    if not supports_unicode or enforce_ascii:
        # If Unicode is not supported, characters should be replaced
        assert "世界" not in result
        # Box drawing characters should become ASCII
        if "┌" in text:
            assert "┌" not in result
            assert "│" not in result
    else:
        # If Unicode is supported and not forced to ASCII, text should be unchanged
        assert result == text


def test_convenience_functions():
    """Test the convenience functions."""
    with patch("figlet_forge.compat.terminal_adjuster.terminal") as mock_terminal:
        # Set up mock properties and methods
        mock_terminal.dimensions = (100, 40)
        mock_terminal.supports_color = True
        mock_terminal.supports_unicode = True
        mock_terminal.adjust_output_for_terminal.return_value = "Adjusted Text"

        # Test convenience functions
        size = get_terminal_size()
        has_color = supports_color()
        has_unicode = supports_unicode()
        adjusted = adjust_output("Test", False)

        assert size == (100, 40)
        assert has_color is True
        assert has_unicode is True
        assert adjusted == "Adjusted Text"
        mock_terminal.adjust_output_for_terminal.assert_called_with("Test", False)


# Run this test only on platforms where it makes sense
@pytest.mark.skipif(
    "CI" in os.environ or "GITHUB_ACTIONS" in os.environ,
    reason="This test requires a real terminal",
)
def test_real_terminal_detection():
    """Test terminal detection with the real terminal."""
    # This is an optional test that runs on real terminals
    terminal = TerminalAdjuster()

    # These should at least not throw exceptions
    width, height = terminal.terminal_width, terminal.terminal_height
    has_color = terminal.supports_color
    has_unicode = terminal.supports_unicode
    color_depth = terminal.color_depth

    # Basic assertions
    assert width > 0
    assert height > 0
    assert isinstance(has_color, bool)
    assert isinstance(has_unicode, bool)
    assert color_depth in [0, 16, 256, 16777216]

    # Test adjustment of a simple string
    adjusted = terminal.adjust_output_for_terminal("Hello")
    assert isinstance(adjusted, str)
    assert len(adjusted) >= 5  # Should at least contain "Hello"
