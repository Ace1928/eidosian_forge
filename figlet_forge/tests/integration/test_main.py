#!/usr/bin/env python

import io
import unittest
from unittest.mock import MagicMock, call, patch

from figlet_forge.cli.main import list_colors, main, parse_args, read_input
from figlet_forge.core.exceptions import FigletError, FontNotFound


class TestParseArgs(unittest.TestCase):
    """Test argument parsing functionality."""

    def test_basic_arguments(self):
        """Test parsing of basic arguments."""
        args = parse_args(["Hello", "World"])
        self.assertEqual(args.text, ["Hello", "World"])
        self.assertFalse(hasattr(args, "version") and args.version)

    def test_font_option(self):
        """Test font option parsing."""
        args = parse_args(["--font", "slant", "Hello"])
        self.assertEqual(args.font, "slant")
        self.assertEqual(args.text, ["Hello"])

    def test_width_option(self):
        """Test width option parsing."""
        args = parse_args(["--width", "100", "Hello"])
        self.assertEqual(args.width, 100)
        self.assertEqual(args.text, ["Hello"])

    def test_justify_option(self):
        """Test justify option parsing."""
        args = parse_args(["--justify", "center", "Hello"])
        self.assertEqual(args.justify, "center")
        self.assertEqual(args.text, ["Hello"])

    def test_direction_option(self):
        """Test direction option parsing."""
        args = parse_args(["--direction", "right-to-left", "Hello"])
        self.assertEqual(args.direction, "right-to-left")
        self.assertEqual(args.text, ["Hello"])

    def test_transformation_options(self):
        """Test transformation options parsing."""
        args = parse_args(
            ["--reverse", "--flip", "--border", "double", "--shade", "Hello"]
        )
        self.assertTrue(args.reverse)
        self.assertTrue(args.flip)
        self.assertEqual(args.border, "double")
        self.assertTrue(args.shade)
        self.assertEqual(args.text, ["Hello"])

    def test_color_option_with_value(self):
        """Test color option with explicit value."""
        args = parse_args(["--color", "RED", "Hello"])
        self.assertEqual(args.color, "RED")
        self.assertEqual(args.text, ["Hello"])

    def test_color_option_as_flag(self):
        """Test color option used as a flag."""
        args = parse_args(["--color", "Hello"])
        self.assertEqual(args.color, "RAINBOW")
        self.assertEqual(args.text, ["Hello"])

    def test_output_options(self):
        """Test output options parsing."""
        args = parse_args(
            ["--unicode", "--output", "output.txt", "--html", "--svg", "Hello"]
        )
        self.assertTrue(args.unicode)
        self.assertEqual(args.output, "output.txt")
        self.assertTrue(args.html)
        self.assertTrue(args.svg)
        self.assertEqual(args.text, ["Hello"])

    def test_showcase_options(self):
        """Test showcase options parsing."""
        args = parse_args(
            [
                "--showcase",
                "--sample-text",
                "Test",
                "--sample-color",
                "RED",
                "--sample-fonts",
                "slant,small",
            ]
        )
        self.assertTrue(args.showcase)
        self.assertEqual(args.sample_text, "Test")
        self.assertEqual(args.sample_color, "RED")
        self.assertEqual(args.sample_fonts, "slant,small")

    def test_showcase_options_as_flags(self):
        """Test showcase options used as flags."""
        args = parse_args(
            ["--showcase", "--sample-text", "--sample-color", "--sample-fonts"]
        )
        self.assertTrue(args.showcase)
        self.assertEqual(args.sample_text, "Hello Eidos")
        self.assertEqual(args.sample_color, "ALL")
        self.assertEqual(args.sample_fonts, "ALL")

    def test_special_format_options(self):
        """Test options with equals sign format."""
        args = parse_args(
            ["--sample=Test", "--sample-color=RED", "--sample-fonts=slant,small"]
        )
        self.assertTrue(args.showcase)
        self.assertEqual(args.sample_text, "Test")
        self.assertEqual(args.sample_color, "RED")
        self.assertEqual(args.sample_fonts, "slant,small")

    def test_version_option(self):
        """Test version option parsing."""
        args = parse_args(["--version"])
        self.assertTrue(args.version)

    def test_list_fonts_option(self):
        """Test list-fonts option parsing."""
        args = parse_args(["--list-fonts"])
        self.assertTrue(args.list_fonts)

    def test_color_list_option(self):
        """Test color-list option parsing."""
        args = parse_args(["--color-list"])
        self.assertTrue(args.color_list)


class TestReadInput(unittest.TestCase):
    """Test reading input from STDIN."""

    @patch("sys.stdin")
    def test_read_from_pipe(self, mock_stdin):
        """Test reading input from pipe."""
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "Hello from pipe\n"

        result = read_input()
        self.assertEqual(result, "Hello from pipe")

    @patch("sys.stdin")
    def test_read_from_tty(self, mock_stdin):
        """Test behavior when no input from pipe."""
        mock_stdin.isatty.return_value = True

        result = read_input()
        self.assertEqual(result, "")


class TestListColors(unittest.TestCase):
    """Test listing available colors."""

    @patch("builtins.print")
    @patch(
        "figlet_forge.cli.main.color_formats",
        {"rainbow": "Rainbow colors", "gradient": "Gradient colors"},
    )
    @patch(
        "figlet_forge.cli.main.COLOR_CODES",
        {"RED": "31", "GREEN": "32", "BLUE": "34", "YELLOW": "33"},
    )
    def test_list_colors(self, mock_print):
        """Test displaying color information."""
        list_colors()

        # Check that print was called with expected information
        mock_print.assert_any_call("Available color names:")
        mock_print.assert_any_call("\nSpecial color formats:")
        mock_print.assert_any_call("  rainbow: Rainbow colors")
        mock_print.assert_any_call("  gradient: Gradient colors")
        mock_print.assert_any_call("\nUsage examples:")


class TestMainFunction(unittest.TestCase):
    """Test the main CLI function."""

    @patch("figlet_forge.cli.main.__version__", "1.0.0")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_version_display(self, mock_stdout):
        """Test displaying version information."""
        result = main(["--version"])

        self.assertEqual(result, 0)
        # Check for version string in either format
        output = mock_stdout.getvalue()
        self.assertTrue(
            "1.0.0" in output,
            f"Version not found in output: {output}"
        )

    @patch("figlet_forge.cli.main.Figlet")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_list_fonts(self, mock_stdout, mock_figlet_class):
        """Test listing available fonts."""
        # Set up mock - use get_fonts (not getFonts)
        mock_figlet = MagicMock()
        mock_figlet.get_fonts.return_value = ["standard", "slant", "small"]
        mock_figlet_class.return_value = mock_figlet

        result = main(["--list-fonts"])

        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn("Available fonts", output)
        self.assertIn("standard", output)
        self.assertIn("slant", output)
        self.assertIn("small", output)

    @patch("figlet_forge.cli.main.list_colors")
    def test_list_colors_flag(self, mock_list_colors):
        """Test listing colors with flag."""
        result = main(["--color-list"])

        self.assertEqual(result, 0)
        mock_list_colors.assert_called_once()

    @patch("figlet_forge.cli.main.list_colors")
    def test_list_colors_option_value(self, mock_list_colors):
        """Test listing colors with option value."""
        result = main(["--color=list"])

        self.assertEqual(result, 0)
        mock_list_colors.assert_called_once()

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_showcase(self, mock_generate_showcase):
        """Test showcase generation."""
        result = main(["--showcase"])

        self.assertEqual(result, 0)
        mock_generate_showcase.assert_called_with(
            sample_text="hello", fonts=None, color=None
        )

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_showcase_with_all_options(self, mock_generate_showcase):
        """Test showcase with all options."""
        result = main(
            [
                "--showcase",
                "--sample-text=Test",
                "--sample-color=RED",
                "--sample-fonts=slant,small",
            ]
        )

        self.assertEqual(result, 0)
        mock_generate_showcase.assert_called_with(
            sample_text="Test", fonts=["slant", "small"], color="RED"
        )

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_showcase_with_all_fonts(self, mock_generate_showcase):
        """Test showcase with all fonts."""
        result = main(["--showcase", "--sample-fonts=ALL"])

        self.assertEqual(result, 0)
        mock_generate_showcase.assert_called_with(
            sample_text="hello", fonts="ALL", color=None
        )

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_showcase_exception(self, mock_generate_showcase):
        """Test exception handling in showcase."""
        mock_generate_showcase.side_effect = Exception("Showcase error")

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            result = main(["--showcase"])

            self.assertEqual(result, 1)
            self.assertIn(
                "Error generating showcase: Showcase error", mock_stderr.getvalue()
            )

    @patch("figlet_forge.cli.main.read_input")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_no_input(self, mock_stderr, mock_read_input):
        """Test behavior with no input."""
        mock_read_input.return_value = ""

        result = main([])

        self.assertEqual(result, 1)
        self.assertIn("No input provided", mock_stderr.getvalue())

    @patch("figlet_forge.cli.main.Figlet")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_basic_text_rendering(self, mock_stdout, mock_figlet_class):
        """Test basic text rendering."""
        # Set up mock
        mock_figlet = MagicMock()
        mock_figlet.renderText.return_value = "Rendered Text"
        mock_figlet_class.return_value = mock_figlet

        result = main(["Hello"])

        self.assertEqual(result, 0)
        mock_figlet.renderText.assert_called_with("Hello")
        self.assertIn("Rendered Text", mock_stdout.getvalue())

    @patch("figlet_forge.cli.main.Figlet")
    @patch("figlet_forge.cli.main.get_terminal_size")
    def test_width_detection(self, mock_get_terminal_size, mock_figlet_class):
        """Test terminal width detection."""
        mock_get_terminal_size.return_value = (80, 25)
        mock_figlet = MagicMock()
        mock_figlet.renderText.return_value = "Rendered Text"
        mock_figlet_class.return_value = mock_figlet

        result = main(["Hello"])

        self.assertEqual(result, 0)
        mock_figlet_class.assert_called_with(
            font="standard",
            width=80,
            justify=None,
            direction=None,
            unicode_aware=False,
        )

    @patch("figlet_forge.cli.main.Figlet")
    def test_transformations(self, mock_figlet_class):
        """Test applying transformations to rendered text."""
        mock_figlet = MagicMock()
        mock_result = MagicMock()
        mock_result.reverse.return_value = mock_result
        mock_result.flip.return_value = mock_result
        mock_result.border.return_value = mock_result
        mock_result.shadow.return_value = mock_result
        mock_result.__str__.return_value = "Transformed Text"

        mock_figlet.renderText.return_value = mock_result
        mock_figlet_class.return_value = mock_figlet

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = main(
                ["--reverse", "--flip", "--border=double", "--shade", "Hello"]
            )

            self.assertEqual(result, 0)
            mock_result.reverse.assert_called_once()
            mock_result.flip.assert_called_once()
            mock_result.border.assert_called_once_with(style="double")
            mock_result.shadow.assert_called_once()
            self.assertIn("Transformed Text", mock_stdout.getvalue())

    @patch("figlet_forge.cli.main.Figlet")
    @patch("figlet_forge.cli.main.colored_format")
    def test_color_formatting(self, mock_colored_format, mock_figlet_class):
        """Test applying color to rendered text."""
        mock_figlet = MagicMock()
        mock_figlet.renderText.return_value = "Rendered Text"
        mock_figlet_class.return_value = mock_figlet

        mock_colored_format.return_value = "Colored Text"

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = main(["--color=RED", "Hello"])

            self.assertEqual(result, 0)
            mock_colored_format.assert_called_with("Rendered Text", "red")
            self.assertIn("Colored Text", mock_stdout.getvalue())

    @patch("figlet_forge.cli.main.Figlet")
    @patch("figlet_forge.cli.main.get_coloring_functions")
    def test_rainbow_color(self, mock_get_coloring_functions, mock_figlet_class):
        """Test applying rainbow coloring."""
        mock_figlet = MagicMock()
        mock_figlet.renderText.return_value = "Rendered Text"
        mock_figlet_class.return_value = mock_figlet

        mock_rainbow = MagicMock(return_value="Rainbow Text")
        mock_get_coloring_functions.return_value = {"rainbow": mock_rainbow}

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = main(["--color=rainbow", "Hello"])

            self.assertEqual(result, 0)
            mock_rainbow.assert_called_with("Rendered Text")
            self.assertIn("Rainbow Text", mock_stdout.getvalue())

    @patch("figlet_forge.cli.main.Figlet")
    @patch("figlet_forge.cli.main.get_coloring_functions")
    def test_gradient_color(self, mock_get_coloring_functions, mock_figlet_class):
        """Test applying gradient coloring."""
        mock_figlet = MagicMock()
        mock_figlet.renderText.return_value = "Rendered Text"
        mock_figlet_class.return_value = mock_figlet

        mock_gradient = MagicMock(return_value="Gradient Text")
        mock_get_coloring_functions.return_value = {"gradient": mock_gradient}

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = main(["--color=red_to_blue", "Hello"])

            self.assertEqual(result, 0)
            mock_gradient.assert_called_with("Rendered Text", "red", "blue")
            self.assertIn("Gradient Text", mock_stdout.getvalue())

    @patch("figlet_forge.cli.main.Figlet")
    def test_html_output(self, mock_figlet_class):
        """Test HTML output formatting."""
        mock_figlet = MagicMock()
        mock_figlet.renderText.return_value = "ASCII Art"
        mock_figlet_class.return_value = mock_figlet

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = main(["--html", "Hello"])

            self.assertEqual(result, 0)
            self.assertIn(
                "<pre style='font-family: monospace;'>ASCII Art</pre>",
                mock_stdout.getvalue(),
            )

    @patch("figlet_forge.cli.main.Figlet")
    def test_svg_output(self, mock_figlet_class):
        """Test SVG output formatting."""
        mock_figlet = MagicMock()
        mock_figlet.renderText.return_value = "ASCII Art"
        mock_figlet_class.return_value = mock_figlet

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = main(["--svg", "Hello"])

            self.assertEqual(result, 0)
            self.assertIn(
                '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
                mock_stdout.getvalue(),
            )
            self.assertIn("<svg", mock_stdout.getvalue())
            self.assertIn("ASCII Art", mock_stdout.getvalue())

    @patch("figlet_forge.cli.main.Figlet")
    def test_file_output(self, mock_figlet_class):
        """Test writing output to file."""
        mock_figlet = MagicMock()
        mock_figlet.renderText.return_value = "ASCII Art"
        mock_figlet_class.return_value = mock_figlet

        mock_open = unittest.mock.mock_open()
        with patch("builtins.open", mock_open):
            result = main(["--output=output.txt", "Hello"])

            self.assertEqual(result, 0)
            mock_open.assert_called_once_with("output.txt", "w", encoding="utf-8")
            handle = mock_open()
            handle.write.assert_has_calls([call("ASCII Art"), call("\n")])

    @patch("figlet_forge.cli.main.Figlet")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_font_not_found_error(self, mock_stderr, mock_figlet_class):
        """Test handling of FontNotFound error."""
        mock_figlet_class.side_effect = FontNotFound("nonexistent_font")

        result = main(["--font=nonexistent_font", "Hello"])

        self.assertEqual(result, 1)
        self.assertIn(
            "Error: Font not found - nonexistent_font", mock_stderr.getvalue()
        )

    @patch("figlet_forge.cli.main.Figlet")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_figlet_error(self, mock_stderr, mock_figlet_class):
        """Test handling of FigletError."""
        mock_figlet_class.side_effect = FigletError("Figlet processing error")

        result = main(["Hello"])

        self.assertEqual(result, 1)
        self.assertIn("Error: Figlet processing error", mock_stderr.getvalue())

    @patch("figlet_forge.cli.main.parse_args")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_keyboard_interrupt(self, mock_stderr, mock_parse_args):
        """Test handling of KeyboardInterrupt."""
        mock_parse_args.side_effect = KeyboardInterrupt()

        result = main([])

        self.assertEqual(result, 130)
        self.assertIn("Operation cancelled by user", mock_stderr.getvalue())

    @patch("figlet_forge.cli.main.parse_args")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_general_exception(self, mock_stderr, mock_parse_args):
        """Test handling of general exceptions."""
        mock_parse_args.side_effect = ValueError("Unexpected error")

        result = main([])

        self.assertEqual(result, 2)
        self.assertIn("Error: Unexpected error", mock_stderr.getvalue())


class TestCLIOptions(unittest.TestCase):
    """Test various CLI options and combinations."""

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_sample_flag(self):
        """Test that --sample works as a flag without requiring value."""
        # Test with just --sample flag
        main(["--sample"])

        # Call arguments validation
        args, kwargs = TestCLIOptions._get_last_call_args(generate_showcase)
        self.assertEqual(kwargs["sample_text"], "hello")
        self.assertIsNone(kwargs["fonts"])
        self.assertIsNone(kwargs["color"])

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_sample_with_text(self):
        """Test sample with explicit sample text."""
        main(["--sample", "--sample-text=Test"])

        args, kwargs = TestCLIOptions._get_last_call_args(generate_showcase)
        self.assertEqual(kwargs["sample_text"], "Test")

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_sample_with_color_and_fonts(self):
        """Test that --sample works with color and font options."""
        main(["--sample", "--sample-color=RED", "--sample-fonts=slant,small"])

        args, kwargs = TestCLIOptions._get_last_call_args(generate_showcase)
        self.assertEqual(kwargs["sample_text"], "hello")
        self.assertEqual(kwargs["fonts"], ["slant", "small"])
        self.assertEqual(kwargs["color"], "RED")

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_sample_color_as_flag(self):
        """Test that --sample-color works as a standalone flag."""
        main(["--sample", "--sample-color"])

        args, kwargs = TestCLIOptions._get_last_call_args(generate_showcase)
        self.assertEqual(kwargs["color"], "ALL")

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_sample_fonts_as_flag(self):
        """Test that --sample-fonts works as a standalone flag."""
        main(["--sample", "--sample-fonts"])

        args, kwargs = TestCLIOptions._get_last_call_args(generate_showcase)
        self.assertEqual(kwargs["fonts"], "ALL")

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_showcase_equivalent_to_sample(self):
        """Test that --showcase is equivalent to --sample."""
        # Test --showcase
        main(["--showcase"])
        args1, kwargs1 = TestCLIOptions._get_last_call_args(generate_showcase)

        generate_showcase.reset_mock()

        # Test --sample
        main(["--sample"])
        args2, kwargs2 = TestCLIOptions._get_last_call_args(generate_showcase)

        # They should be the same
        self.assertEqual(kwargs1, kwargs2)

    @patch("figlet_forge.cli.main.colored_format")
    @patch("figlet_forge.cli.main.get_coloring_functions")
    def test_color_flag_without_value(self, mock_get_coloring, mock_colored_format):
        """Test that --color works as a flag without requiring value."""
        # Set up mock for get_coloring_functions to return a mock rainbow function
        mock_rainbow = unittest.mock.MagicMock(return_value="Rainbow text")
        mock_get_coloring.return_value = {"rainbow": mock_rainbow}

        with patch("figlet_forge.cli.main.Figlet") as mock_figlet_class:
            mock_figlet = MagicMock()
            mock_figlet.renderText.return_value = "Rendered text"
            mock_figlet_class.return_value = mock_figlet

            # Use --color as a flag (should default to rainbow)
            main(["--color", "Hello"])

            # The rainbow colorizer should have been called
            mock_rainbow.assert_called_once()

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_sample_text_without_value(self):
        """Test that --sample-text works without a value."""
        main(["--sample", "--sample-text"])

        args, kwargs = TestCLIOptions._get_last_call_args(generate_showcase)
        self.assertEqual(kwargs["sample_text"], "Hello Eidos")

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_special_formatting_options(self):
        """Test special formatting options like equals sign."""
        # Test special formatting with equals sign
        main(["--sample=Custom Text"])

        args1, kwargs1 = TestCLIOptions._get_last_call_args(generate_showcase)
        self.assertEqual(kwargs1["sample_text"], "Custom Text")

        generate_showcase.reset_mock()

        # Test combination of special formats
        main(["--showcase=Test", "--sample-color=RED"])

        args2, kwargs2 = TestCLIOptions._get_last_call_args(generate_showcase)
        self.assertEqual(kwargs2["sample_text"], "Test")
        self.assertEqual(kwargs2["color"], "RED")

    @patch("figlet_forge.cli.main.generate_showcase")
    def test_complex_option_combinations(self):
        """Test complex combinations of options."""
        main(
            [
                "--sample",
                "--sample-text=Complex",
                "--sample-color=ALL",
                "--sample-fonts=ALL",
            ]
        )

        args, kwargs = TestCLIOptions._get_last_call_args(generate_showcase)
        self.assertEqual(kwargs["sample_text"], "Complex")
        self.assertEqual(kwargs["color"], "ALL")
        self.assertEqual(kwargs["fonts"], "ALL")

    @staticmethod
    def _get_last_call_args(mock_function):
        """Helper to get the last call arguments from a mock function."""
        mock_calls = mock_function.mock_calls
        if not mock_calls:
            return (), {}
        # Get the last call and extract args and kwargs
        name, args, kwargs = mock_calls[-1]
        return args, kwargs


if __name__ == "__main__":
    unittest.main()
