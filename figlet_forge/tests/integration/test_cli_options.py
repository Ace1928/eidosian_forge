"""
Integrated tests for CLI option combinations.

Tests complex combinations of command-line arguments and option interactions
to ensure robust handling of all possible user inputs.
"""

import io
import os
from unittest.mock import MagicMock, patch

import pytest

from figlet_forge.cli.main import main


class TestOptionCombinations:
    """Test combinations of command-line options."""

    @pytest.mark.parametrize(
        "options",
        [
            # Basic options
            ["Hello"],
            ["--font=slant", "Hello"],
            ["--width=120", "Hello"],
            ["--justify=center", "Hello"],
            ["--direction=right-to-left", "Hello"],
            # Transformation options
            ["--reverse", "Hello"],
            ["--flip", "Hello"],
            ["--border=single", "Hello"],
            ["--shade", "Hello"],
            # Color options
            ["--color=RED", "Hello"],
            ["--color", "Hello"],  # Default to rainbow
            # Output format options
            ["--output=temp_test_output.txt", "Hello"],
            ["--html", "Hello"],
            ["--svg", "Hello"],
            ["--unicode", "Hello"],
            # Option combinations
            ["--font=slant", "--color=RED", "--justify=center", "Hello"],
            ["--reverse", "--flip", "--border=double", "--shade", "Hello"],
            ["--html", "--unicode", "Hello"],
            # Showcase options
            ["--sample"],
            ["--showcase"],
            ["--sample-text=Test"],
            ["--sample-color=RED"],
            ["--sample-fonts=slant,small"],
            # Combination of showcase options
            ["--sample", "--sample-text=Test", "--sample-color=RED"],
            ["--showcase", "--sample-fonts=slant,small"],
            ["--sample-text", "--sample-color", "--sample-fonts"],
        ],
    )
    def test_option_combinations(self, options):
        """Test various combinations of command-line options."""
        with patch("sys.stdout", new_callable=io.StringIO), patch(
            "figlet_forge.cli.main.Figlet"
        ) as mock_figlet_class:
            # Set up mock to handle render calls
            mock_figlet = MagicMock()
            mock_result = MagicMock()
            mock_result.__str__.return_value = "Rendered Text"

            # Handle transformations
            mock_result.reverse.return_value = mock_result
            mock_result.flip.return_value = mock_result
            mock_result.border.return_value = mock_result
            mock_result.shadow.return_value = mock_result

            mock_figlet.renderText.return_value = mock_result
            mock_figlet_class.return_value = mock_figlet

            # For file output tests, mock open
            if any("--output=" in opt for opt in options):
                with patch("builtins.open", unittest.mock.mock_open()) as mock_open:
                    result = main(options)
                    assert result == 0
            else:
                result = main(options)
                assert result == 0

    @pytest.mark.parametrize(
        "options,error_expected",
        [
            # Invalid font
            (["--font=nonexistent", "Hello"], True),
            # Invalid color
            (["--color=NONEXISTENT", "Hello"], True),
            # Invalid border
            (["--border=nonexistent", "Hello"], True),
            # No input
            ([], True),
            # Invalid combination
            (["--html", "--svg"], True),  # Cannot have both HTML and SVG
        ],
    )
    def test_error_handling(self, options, error_expected):
        """Test error handling with invalid options."""
        with patch("sys.stderr", new_callable=io.StringIO), patch(
            "figlet_forge.cli.main.Figlet"
        ) as mock_figlet_class:
            # Simulate errors for specific cases
            if "--font=nonexistent" in " ".join(options):
                mock_figlet_class.side_effect = FontNotFound("nonexistent")

            # For normal execution
            mock_figlet = MagicMock()
            mock_figlet.renderText.return_value = "Rendered Text"
            mock_figlet_class.return_value = mock_figlet

            result = main(options)

            if error_expected:
                assert result != 0  # Non-zero exit code indicates error
            else:
                assert result == 0


class TestSpecialArgForms:
    """Test special argument formats and behaviors."""

    @pytest.mark.parametrize(
        "arg_format,expected_value",
        [
            ("--color", "RAINBOW"),  # Flag form defaults to rainbow
            ("--color=RED", "RED"),  # Equals form sets explicit value
            ("--color RED", "RED"),  # Space form sets explicit value
        ],
    )
    def test_color_arg_formats(self, arg_format, expected_value):
        """Test different formats for the color argument."""
        with patch("sys.stdout", new_callable=io.StringIO), patch(
            "figlet_forge.cli.main.parse_args"
        ) as mock_parse_args:
            # Set up args based on format
            if " " in arg_format:
                options = arg_format.split()
                options.append("Hello")  # Add text
            else:
                options = [arg_format, "Hello"]

            main(options)

            # Check how args was constructed
            args, _ = mock_parse_args.call_args
            assert args[0] == options

    @pytest.mark.parametrize(
        "showcase_args,expected_text,expected_color,expected_fonts",
        [
            # Basic showcase
            (["--showcase"], "hello", None, None),
            # With explicit text
            (["--showcase", "--sample-text=Test"], "Test", None, None),
            # With color
            (["--showcase", "--sample-color=RED"], "hello", "RED", None),
            # With fonts
            (
                ["--showcase", "--sample-fonts=slant,small"],
                "hello",
                None,
                ["slant", "small"],
            ),
            # With all options
            (
                [
                    "--showcase",
                    "--sample-text=Test",
                    "--sample-color=RED",
                    "--sample-fonts=slant,small",
                ],
                "Test",
                "RED",
                ["slant", "small"],
            ),
            # Flag forms (without values)
            (["--sample", "--sample-text"], "Hello Eidos", None, None),
            (["--sample", "--sample-color"], "hello", "ALL", None),
            (["--sample", "--sample-fonts"], "hello", None, "ALL"),
        ],
    )
    def test_showcase_options(
        self, showcase_args, expected_text, expected_color, expected_fonts
    ):
        """Test different showcase option combinations."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            main(showcase_args)

            # Check the call to generate_showcase
            mock_showcase.assert_called_once()
            _, kwargs = mock_showcase.call_args

            assert kwargs["sample_text"] == expected_text
            assert kwargs["color"] == expected_color
            assert kwargs["fonts"] == expected_fonts

    @pytest.mark.parametrize(
        "source_args,expected_text",
        [
            # Direct command line text
            (["Hello World"], "Hello World"),
            # Multiple words
            (["Hello", "World"], "Hello World"),
        ],
    )
    def test_text_sources(self, source_args, expected_text):
        """Test different text input sources."""
        with patch("sys.stdout", new_callable=io.StringIO), patch(
            "figlet_forge.cli.main.Figlet"
        ) as mock_figlet_class:
            mock_figlet = MagicMock()
            mock_figlet.renderText.return_value = "Rendered Text"
            mock_figlet_class.return_value = mock_figlet

            main(source_args)

            # Check that text was properly joined
            mock_figlet.renderText.assert_called_with(expected_text)


# Create a temporary test file that can be removed when done
@pytest.fixture
def temp_output_file():
    filename = "temp_test_output.txt"
    yield filename
    if os.path.exists(filename):
        os.remove(filename)


class TestOutputRedirection:
    """Test output redirection options."""

    def test_file_output(self, temp_output_file):
        """Test writing output to a file."""
        with patch("figlet_forge.cli.main.Figlet") as mock_figlet_class:
            mock_figlet = MagicMock()
            mock_figlet.renderText.return_value = "Rendered Text"
            mock_figlet_class.return_value = mock_figlet

            result = main([f"--output={temp_output_file}", "Hello"])

            assert result == 0
            assert os.path.exists(temp_output_file)
            with open(temp_output_file, encoding="utf-8") as f:
                content = f.read()
                assert "Rendered Text" in content

    def test_html_output(self):
        """Test HTML output format."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout, patch(
            "figlet_forge.cli.main.Figlet"
        ) as mock_figlet_class:
            mock_figlet = MagicMock()
            mock_figlet.renderText.return_value = "Rendered Text"
            mock_figlet_class.return_value = mock_figlet

            result = main(["--html", "Hello"])

            assert result == 0
            output = mock_stdout.getvalue()
            assert "<pre" in output
            assert "Rendered Text" in output
            assert "</pre>" in output

    def test_svg_output(self):
        """Test SVG output format."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout, patch(
            "figlet_forge.cli.main.Figlet"
        ) as mock_figlet_class:
            mock_figlet = MagicMock()
            mock_figlet.renderText.return_value = "Rendered Text"
            mock_figlet_class.return_value = mock_figlet

            result = main(["--svg", "Hello"])

            assert result == 0
            output = mock_stdout.getvalue()
            assert "<?xml" in output
            assert "<svg" in output
            assert "Rendered Text" in output
