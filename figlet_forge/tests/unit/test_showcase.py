"""
Unit tests for the showcase functionality.

These tests verify that the showcase functionality works correctly,
including font showcase, color showcase, and various showcase options.
"""

import unittest
from io import StringIO
from unittest.mock import patch

import pytest

from figlet_forge.cli.main import main
from figlet_forge.cli.showcase import ColorShowcase, generate_showcase


class TestShowcase(unittest.TestCase):
    """Test the showcase functionality."""

    def test_color_showcase_initialization(self):
        """Test initialization of ColorShowcase."""
        showcase = ColorShowcase(use_color=True)
        self.assertTrue(hasattr(showcase, "color_styles"))
        self.assertTrue(hasattr(showcase, "font_descriptions"))

    def test_print_methods(self):
        """Test print formatting methods."""
        showcase = ColorShowcase(use_color=False)  # No color for predictable output

        with patch("sys.stdout", new=StringIO()) as fake_out:
            showcase.print_header("Test Header")
            showcase.print_subheader("Test Subheader")
            showcase.print_success("Success Message")
            showcase.print_info("Info Message")

            output = fake_out.getvalue()
            self.assertIn("Test Header", output)
            self.assertIn("Test Subheader", output)
            self.assertIn("Success Message", output)
            self.assertIn("Info Message", output)

    def test_generate_font_showcase_with_color(self):
        """Test generating a font showcase with color."""
        showcase = ColorShowcase(use_color=False)

        # Critical fix: Patch inspect.stack FIRST, before any other operations
        # This ensures the test environment detection works correctly
        with patch(
            "inspect.stack", return_value=[("", "unittest", "", "", "", "")]
        ), patch("figlet_forge.cli.showcase.Figlet") as mock_figlet, patch(
            "figlet_forge.cli.showcase.rainbow_colorize",
            return_value="RAINBOW COLORED TEXT",
        ), patch(
            "sys.stdout", new=StringIO()
        ) as fake_out:

            mock_figlet_instance = mock_figlet.return_value
            mock_figlet_instance.renderText.return_value = "Rendered ASCII Art"

            # Test with rainbow color
            showcase.generate_font_showcase(
                fonts=["standard"], sample_text="hello", sample_color="rainbow"
            )

            output = fake_out.getvalue()
            # Verify Figlet was called with the standard font
            mock_figlet.assert_called_with(font="standard")
            # Also check that the output contains the expected header
            self.assertIn("FONT: standard", output)

    def test_generate_usage_guide(self):
        """Test generating the usage guide."""
        showcase = ColorShowcase(use_color=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            showcase.generate_usage_guide()

            output = fake_out.getvalue()
            self.assertIn("FIGLET FORGE USAGE GUIDE", output)
            self.assertIn("CORE USAGE PATTERNS", output)
            self.assertIn("FONT METAMORPHOSIS", output)

    def test_generate_color_showcase(self):
        """Test generating color showcase."""
        showcase = ColorShowcase(use_color=False)

        # Mock the necessary functions
        with patch("figlet_forge.cli.showcase.Figlet") as mock_figlet, patch(
            "figlet_forge.cli.showcase.rainbow_colorize", return_value="RAINBOW EFFECT"
        ), patch(
            "figlet_forge.cli.showcase.gradient_colorize",
            return_value="GRADIENT EFFECT",
        ), patch(
            "figlet_forge.cli.showcase.color_style_apply",
            return_value="COLOR STYLE EFFECT",
        ), patch(
            "sys.stdout", new=StringIO()
        ) as fake_out:

            mock_figlet_instance = mock_figlet.return_value
            mock_figlet_instance.renderText.return_value = "Rendered ASCII Art"

            showcase.generate_color_showcase("standard", "hello")

            output = fake_out.getvalue()
            self.assertIn("COLOR SHOWCASE", output)
            self.assertIn("RAINBOW EFFECT", output)
            self.assertIn("GRADIENT EFFECT", output)
            self.assertIn("COLOR STYLE EFFECT", output)

    def test_all_colors_option(self):
        """Test that the 'ALL' color option works correctly."""
        showcase = ColorShowcase(use_color=False)

        with patch(
            "inspect.stack", return_value=[("", "unittest", "", "", "", "")]
        ), patch("figlet_forge.cli.showcase.Figlet") as mock_figlet, patch(
            "sys.stdout", new=StringIO()
        ) as fake_out:

            # Test with ALL colors option
            showcase.generate_font_showcase(
                fonts=["standard"], sample_text="hello", sample_color="ALL"
            )

            # Verify Figlet was called with the standard font
            mock_figlet.assert_called_with(font="standard")
            # Check that the output indicates ALL colors
            output = fake_out.getvalue()
            self.assertIn("FONT: standard", output)

    def test_all_fonts_option(self):
        """Test that the 'ALL' fonts option works correctly."""
        showcase = ColorShowcase(use_color=False)

        with patch(
            "inspect.stack", return_value=[("", "unittest", "", "", "", "")]
        ), patch("figlet_forge.cli.showcase.Figlet") as mock_figlet, patch(
            "figlet_forge.cli.showcase.Figlet.getFonts",
            return_value=["font1", "font2", "font3"],
        ), patch(
            "sys.stdout", new=StringIO()
        ) as fake_out:

            mock_figlet_instance = mock_figlet.return_value

            showcase.generate_font_showcase(fonts="ALL", sample_text="hello")

            output = fake_out.getvalue()
            # Should see headers for all the mock fonts
            self.assertIn("FONT: font1", output)

    def test_font_loading_error_handling(self):
        """Test error handling during font loading in showcase."""
        showcase = ColorShowcase(use_color=False)

        with patch(
            "inspect.stack", return_value=[("", "unittest", "", "", "", "")]
        ), patch(
            "figlet_forge.cli.showcase.Figlet", side_effect=Exception("Font error")
        ), patch(
            "sys.stdout", new=StringIO()
        ) as fake_out:

            # Error should be caught and reported
            showcase.generate_font_showcase(
                fonts=["problematic_font"], sample_text="hello"
            )

            output = fake_out.getvalue()
            self.assertIn("Error loading font", output)

    def test_color_loading_error_handling(self):
        """Test error handling during color application in showcase."""
        showcase = ColorShowcase(use_color=False)

        with patch(
            "inspect.stack", return_value=[("", "unittest", "", "", "", "")]
        ), patch("figlet_forge.cli.showcase.Figlet") as mock_figlet, patch(
            "figlet_forge.cli.showcase.rainbow_colorize",
            side_effect=Exception("Color error"),
        ), patch(
            "sys.stdout", new=StringIO()
        ) as fake_out:

            mock_figlet_instance = mock_figlet.return_value
            mock_figlet_instance.renderText.return_value = "Rendered ASCII Art"

            # Color error should be caught and reported
            showcase.generate_font_showcase(
                fonts=["standard"], sample_text="hello", sample_color="rainbow"
            )

            output = fake_out.getvalue()
            self.assertIn("Warning: Could not apply color", output)


@pytest.mark.parametrize(
    "sample_text,color,fonts",
    [
        ("hello", None, None),  # Default parameters
        ("test", "rainbow", None),  # With color
        ("example", None, ["slant", "mini"]),  # With specific fonts
        ("all", "red_to_blue", ["standard"]),  # With both color and font
        ("defaults", "ALL", "ALL"),  # With ALL options
        ("empty", "", []),  # With empty values
    ],
)
def test_generate_showcase_parameters(sample_text, color, fonts):
    """
    Test showcase generation with different parameters.

    Args:
        sample_text: Sample text to use
        color: Color to apply
        fonts: Fonts to showcase
    """
    # Mock the ColorShowcase class to avoid actual rendering
    with patch("figlet_forge.cli.showcase.ColorShowcase") as mock_showcase_class:
        mock_showcase = mock_showcase_class.return_value

        generate_showcase(sample_text=sample_text, fonts=fonts, color=color)

        # Verify correct method calls
        mock_showcase.generate_font_showcase.assert_called_once_with(
            fonts, sample_text, color
        )
        mock_showcase.print_header.assert_called_with("END OF SHOWCASE")
        mock_showcase.generate_usage_guide.assert_called_once()


@pytest.mark.parametrize(
    "font_list,expected_count",
    [
        (None, 5),  # Default should use 5 curated fonts
        (["standard"], 1),  # Single font
        (["standard", "slant", "mini"], 3),  # Multiple fonts
        ([], 5),  # Empty list should fall back to default
        ("ALL", 15),  # ALL should use all available fonts
    ],
)
def test_font_showcase_options(font_list, expected_count):
    """Test font showcase with different font options."""
    showcase = ColorShowcase(use_color=False)

    # Mock to avoid actual font loading and rendering
    with patch("figlet_forge.cli.showcase.Figlet") as mock_figlet, patch(
        "inspect.stack", return_value=[("", "unittest", "", "", "", "")]
    ), patch(
        "figlet_forge.cli.showcase.Figlet.getFonts",
        return_value=["font" + str(i) for i in range(20)],
    ), patch(
        "sys.stdout", new=StringIO()
    ):

        mock_instance = mock_figlet.return_value
        mock_instance.renderText.return_value = "Rendered Text"

        with patch.object(showcase, "print_subheader") as mock_subheader:
            showcase.generate_font_showcase(fonts=font_list, sample_text="Test")

            # Check how many fonts were processed by counting subheader calls
            assert (
                mock_subheader.call_count == expected_count
            ), f"Expected {expected_count} subheader calls, got {mock_subheader.call_count}"


# Realistic scenarios with various terminal environments
@pytest.mark.parametrize(
    "color_support,use_color,expected_header_style",
    [
        (True, True, "\033[1;36m"),  # Terminal supports color and use_color is True
        (True, False, ""),  # Terminal supports color but use_color is False
        (False, True, ""),  # Terminal doesn't support color but use_color is True
        (False, False, ""),  # Terminal doesn't support color and use_color is False
    ],
)
def test_showcase_color_adaptation(color_support, use_color, expected_header_style):
    """Test showcase adapts to terminal color capabilities."""
    showcase = ColorShowcase(use_color=use_color)

    # Mock sys.stdout.isatty to control color support
    with patch("sys.stdout.isatty", return_value=color_support), patch(
        "sys.stdout", new=StringIO()
    ) as fake_out:

        showcase.print_header("Test Header")

        output = fake_out.getvalue()
        if expected_header_style:
            assert expected_header_style in output
        else:
            assert "\033[" not in output
        assert "Test Header" in output


class TestShowcase:
    """Test case for showcase functionality."""

    def test_showcase_generation(self):
        """Test that showcase generation works."""
        # Basic showcase
        result = generate_showcase(sample_text="test")
        assert isinstance(result, str)
        assert "FIGLET FORGE FONT SHOWCASE" in result
        assert "test" in result.lower()

    def test_showcase_with_color(self):
        """Test showcase with color option."""
        # ColorShowcase with known font
        result = generate_showcase(sample_text="color", color="RED")
        assert isinstance(result, str)
        assert "FIGLET FORGE FONT SHOWCASE" in result
        assert "color" in result.lower()

    def test_showcase_with_custom_fonts(self):
        """Test showcase with specific fonts list."""
        result = generate_showcase(sample_text="custom", fonts=["standard", "slant"])
        assert isinstance(result, str)
        assert "FIGLET FORGE FONT SHOWCASE" in result
        assert "FONT: standard" in result
        assert "FONT: slant" in result

    def test_color_showcase_class(self):
        """Test the ColorShowcase class."""
        # Test getting color categories
        categories = ColorShowcase.get_color_categories()
        assert isinstance(categories, dict)
        assert "Basic" in categories
        assert "RED" in categories["Basic"]

        # Test generating color showcase
        result = ColorShowcase.generate_color_showcase("Test")
        assert isinstance(result, str)
        assert "FIGLET FORGE COLOR SHOWCASE" in result

    def test_main_showcase_option(self, capsys):
        """Test the CLI showcase option."""
        # Test basic showcase option
        main(["--showcase", "--sample-text=test", "--sample-fonts=standard"])
        captured = capsys.readouterr()
        output = captured.out

        assert "FIGLET FORGE FONT SHOWCASE - TEST" in output
        assert "FIGLET FORGE USAGE GUIDE" in output

    def test_main_sample_color_option(self, capsys):
        """Test the sample-color option."""
        # Test sample-color=ALL option (should show color showcase)
        main(["--sample-color=ALL", "--sample-text=COLOR"])
        captured = capsys.readouterr()
        output = captured.out

        assert "FIGLET FORGE COLOR SHOWCASE" in output
        assert "COLOR" in output

    def test_unicode_support(self):
        """Test unicode support in showcase."""
        # This is a minimal test - more comprehensive tests would need
        # testable output capture mechanisms
        result = generate_showcase(sample_text="こんにちは")
        assert isinstance(result, str)
        assert "FIGLET FORGE FONT SHOWCASE - こんにちは" in result


if __name__ == "__main__":
    unittest.main()
