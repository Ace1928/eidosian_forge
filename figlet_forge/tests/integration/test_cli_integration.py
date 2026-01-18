"""
Integration tests for the command-line interface.

These tests verify that the CLI components work together correctly and
handle various input parameters and options appropriately.
"""

import os
import sys
import tempfile
from io import StringIO
from typing import List, Tuple
from unittest.mock import patch

import pytest

from figlet_forge.cli.main import main
from figlet_forge.core.exceptions import FontNotFound


@pytest.fixture
def cli_runner():
    """
    Fixture to run CLI commands with captured output.

    Returns:
        Function to run CLI commands
    """

    def _run(args: List[str]) -> Tuple[int, str, str]:
        """Run CLI with args and return exit code, stdout, and stderr."""
        stdout = StringIO()
        stderr = StringIO()

        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            try:
                exit_code = main(args)
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 1

        return exit_code, stdout.getvalue(), stderr.getvalue()

    return _run


def test_basic_text_rendering(cli_runner):
    """Test basic text rendering from the CLI."""
    exit_code, stdout, stderr = cli_runner(["Hello"])

    assert exit_code == 0
    assert stdout.strip()  # Should have some output
    assert not stderr.strip()  # Should have no errors


def test_font_option(cli_runner):
    """Test the --font option."""
    exit_code, stdout, stderr = cli_runner(["--font=standard", "Test"])

    assert exit_code == 0
    assert stdout.strip()  # Should have some output
    assert not stderr.strip()  # Should have no errors


def test_color_option(cli_runner):
    """Test the --color option."""
    exit_code, stdout, stderr = cli_runner(["--color=RED", "Test"])

    assert exit_code == 0
    assert "\033[" in stdout  # Should contain ANSI color codes
    assert not stderr.strip()  # Should have no errors


def test_list_fonts(cli_runner):
    """Test the --list-fonts option."""
    exit_code, stdout, stderr = cli_runner(["--list-fonts"])

    assert exit_code == 0
    assert "Available fonts:" in stdout
    assert "standard" in stdout  # Should list standard font
    assert not stderr.strip()  # Should have no errors


def test_version_option(cli_runner):
    """Test the --version option."""
    exit_code, stdout, stderr = cli_runner(["--version"])

    assert exit_code == 0
    assert "Figlet Forge version" in stdout
    assert not stderr.strip()  # Should have no errors


def test_invalid_font(cli_runner):
    """Test with an invalid font."""
    # We need to patch FigletFont.__init__ rather than the class itself
    # to ensure the exception is properly raised and handled
    with patch(
        "figlet_forge.core.figlet_font.FigletFont.__init__",
        side_effect=FontNotFound("Font not found: nonexistent"),
    ):
        exit_code, stdout, stderr = cli_runner(["--font=nonexistent", "Test"])

        assert exit_code == 1  # Should fail
        assert "Error:" in stderr  # Should have error message
        assert "Font not found" in stderr


def test_showcase_option(cli_runner):
    """Test the --showcase option."""
    # Mock showcase generation to avoid large output
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--showcase"])

        assert exit_code == 0
        assert not stderr.strip()  # Should have no errors
        mock_showcase.assert_called_once()


def test_sample_flag(cli_runner):
    """Test the --sample flag (equivalent to --showcase)."""
    # Mock showcase generation to avoid large output
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample"])

        assert exit_code == 0
        assert not stderr.strip()  # Should have no errors
        mock_showcase.assert_called_once()


def test_sample_text_option(cli_runner):
    """Test the --sample-text option."""
    # Mock showcase generation to avoid large output
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-text=Custom"])

        assert exit_code == 0
        mock_showcase.assert_called_with(sample_text="Custom", fonts=None, color=None)

    # Test with no value (should default to "Hello Eidos")
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-text"])

        assert exit_code == 0
        mock_showcase.assert_called_with(
            sample_text="Hello Eidos", fonts=None, color=None
        )


def test_sample_color_option(cli_runner):
    """Test the --sample-color option."""
    # Mock showcase generation to avoid large output
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-color=RED"])

        assert exit_code == 0
        mock_showcase.assert_called_with(sample_text="hello", fonts=None, color="RED")

    # Test with no value (should default to ALL)
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-color"])

        assert exit_code == 0
        mock_showcase.assert_called_with(sample_text="hello", fonts=None, color="ALL")


def test_sample_fonts_option(cli_runner):
    """Test the --sample-fonts option."""
    # Mock showcase generation to avoid large output
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(
            ["--sample", "--sample-fonts=slant,mini"]
        )

        assert exit_code == 0
        mock_showcase.assert_called_with(
            sample_text="hello", fonts=["slant", "mini"], color=None
        )

    # Test with no value (should default to ALL)
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-fonts"])

        assert exit_code == 0
        mock_showcase.assert_called_with(sample_text="hello", fonts="ALL", color=None)


def test_sample_color_option(cli_runner):
    """Test the --sample-color option with and without values."""
    # Test with specific color
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-color=RED"])

        assert exit_code == 0
        mock_showcase.assert_called_with(sample_text="hello", fonts=None, color="RED")

    # Test with no value (should default to ALL)
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-color"])

        assert exit_code == 0
        mock_showcase.assert_called_with(sample_text="hello", fonts=None, color="ALL")


def test_sample_fonts_option(cli_runner):
    """Test the --sample-fonts option with and without values."""
    # Test with specific fonts
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(
            ["--sample", "--sample-fonts=slant,mini"]
        )

        assert exit_code == 0
        mock_showcase.assert_called_with(
            sample_text="hello", fonts=["slant", "mini"], color=None
        )

    # Test with no value (should default to ALL)
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-fonts"])

        assert exit_code == 0
        mock_showcase.assert_called_with(sample_text="hello", fonts="ALL", color=None)


def test_sample_text_option(cli_runner):
    """Test the --sample-text option with and without values."""
    # Test with specific text
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-text=Custom"])

        assert exit_code == 0
        mock_showcase.assert_called_with(sample_text="Custom", fonts=None, color=None)

    # Test with no value (should default to "Hello Eidos")
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(["--sample", "--sample-text"])

        assert exit_code == 0
        mock_showcase.assert_called_with(
            sample_text="Hello Eidos", fonts=None, color=None
        )


def test_combined_sample_options(cli_runner):
    """Test combined sample options."""
    # Mock showcase generation to avoid large output
    with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
        exit_code, stdout, stderr = cli_runner(
            [
                "--sample",
                "--sample-text=Hello",
                "--sample-color=RED",
                "--sample-fonts=slant,mini",
            ]
        )

        assert exit_code == 0
        mock_showcase.assert_called_with(
            sample_text="Hello", fonts=["slant", "mini"], color="RED"
        )


def test_transform_options(cli_runner):
    """Test transformation options."""
    # Test reverse
    exit_code1, stdout1, stderr1 = cli_runner(["--reverse", "A"])
    assert exit_code1 == 0

    # Test flip
    exit_code2, stdout2, stderr2 = cli_runner(["--flip", "A"])
    assert exit_code2 == 0

    # Test border
    exit_code3, stdout3, stderr3 = cli_runner(["--border=single", "A"])
    assert exit_code3 == 0
    assert "┌" in stdout3  # Should have border characters

    # Test shadow
    exit_code4, stdout4, stderr4 = cli_runner(["--shade", "A"])
    assert exit_code4 == 0


def test_output_to_file(cli_runner):
    """Test writing output to a file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Run command with output to file
        exit_code, stdout, stderr = cli_runner(["--output", temp_path, "Test"])

        assert exit_code == 0
        assert not stderr.strip()  # No errors

        # Check that file was created with content
        assert os.path.exists(temp_path)
        with open(temp_path) as f:
            content = f.read()
            assert content  # File should not be empty

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_unicode_support(cli_runner):
    """Test Unicode support."""
    # Only run if terminal supports Unicode
    if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower():
        exit_code, stdout, stderr = cli_runner(["--unicode", "世界"])

        assert exit_code == 0
        assert stdout.strip()  # Should have some output
        assert not stderr.strip()  # Should have no errors


def test_html_output(cli_runner):
    """Test HTML output format."""
    exit_code, stdout, stderr = cli_runner(["--html", "Test"])

    assert exit_code == 0
    assert "<pre" in stdout  # Should contain HTML tags
    assert not stderr.strip()


def test_svg_output(cli_runner):
    """Test SVG output format."""
    exit_code, stdout, stderr = cli_runner(["--svg", "Test"])

    assert exit_code == 0
    assert "<svg" in stdout  # Should contain SVG tags
    assert not stderr.strip()


@pytest.mark.parametrize(
    "option_str,expected_exit_code",
    [
        # Valid combinations
        ("--font=standard --color=RED", 0),
        ("--font=standard --reverse --flip", 0),
        ("--font=standard --border=single --shade", 0),
        ("--font=standard --color=rainbow --border=double", 0),
        ("--width=120 --justify=center", 0),
        ("--direction=right-to-left --color=BLUE", 0),
        # Sample options with and without values
        ('--sample --sample-text="Hello World"', 0),
        ("--sample --sample-text", 0),  # No value
        ("--sample --sample-color=RED", 0),
        ("--sample --sample-color", 0),  # No value
        ("--sample --sample-fonts=slant,mini", 0),
        ("--sample --sample-fonts", 0),  # No value
        # Combined sample options
        ('--sample --sample-text="Test" --sample-color=RED', 0),
        (
            "--sample --sample-text --sample-color --sample-fonts",
            0,
        ),  # All without values
        # Color option as flag
        ("--color", 0),  # Should default to rainbow
        # Invalid combinations should still parse correctly
        ("--font=nonexistent", 1),  # Will fail later but parse correctly
    ],
)
def test_option_combinations(cli_runner, option_str, expected_exit_code):
    """Test various combinations of command line options."""
    # Split option string into a list
    options = option_str.split()

    # Add a default text if not using --sample
    if not any(opt.startswith("--sample") for opt in options):
        options.append("Test")

    # For failing cases, we'll mock to make test deterministic
    if expected_exit_code != 0 and "--font=nonexistent" in option_str:
        with patch(
            "figlet_forge.figlet.FigletFont", side_effect=FontNotFound("Font not found")
        ):
            exit_code, stdout, stderr = cli_runner(options)
            assert exit_code == expected_exit_code
    else:
        # For showcase/sample options, mock the showcase generation
        if any(opt == "--sample" for opt in options):
            with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
                exit_code, stdout, stderr = cli_runner(options)
                assert exit_code == expected_exit_code
                mock_showcase.assert_called_once()
        else:
            # For normal options
            exit_code, stdout, stderr = cli_runner(options)
            assert exit_code == expected_exit_code
