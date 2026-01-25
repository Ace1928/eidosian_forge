import unittest
from unittest.mock import patch

from figlet_forge.cli.main import main


class TestCLIOptions(unittest.TestCase):
    """Test CLI options and flags."""

    def test_sample_flag(self):
        """Test that --sample works as a flag without requiring value."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            # Test with just --sample flag
            main(["--sample"])
            mock_showcase.assert_called_with(
                sample_text="hello", fonts=None, color=None
            )

            # Reset mock
            mock_showcase.reset_mock()

            # Test with --sample and --sample-text
            main(["--sample", "--sample-text=Test"])
            mock_showcase.assert_called_with(sample_text="Test", fonts=None, color=None)

    def test_sample_with_color_and_fonts(self):
        """Test that --sample works with color and font options."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            main(["--sample", "--sample-color=RED", "--sample-fonts=slant,small"])
            mock_showcase.assert_called_with(
                sample_text="hello", fonts=["slant", "small"], color="RED"
            )

    def test_sample_color_as_flag(self):
        """Test that --sample-color works as a standalone flag."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            main(["--sample", "--sample-color"])
            mock_showcase.assert_called_with(
                sample_text="hello", fonts=None, color="ALL"
            )

    def test_sample_fonts_as_flag(self):
        """Test that --sample-fonts works as a standalone flag."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            main(["--sample", "--sample-fonts"])
            mock_showcase.assert_called_with(
                sample_text="hello", fonts="ALL", color=None
            )

    def test_showcase_equivalent_to_sample(self):
        """Test that --showcase is equivalent to --sample."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            # Test --showcase
            main(["--showcase"])
            mock_showcase.assert_called_with(
                sample_text="hello", fonts=None, color=None
            )

            # Reset mock
            mock_showcase.reset_mock()

            # Test --sample
            main(["--sample"])
            mock_showcase.assert_called_with(
                sample_text="hello", fonts=None, color=None
            )

    def test_color_flag_without_value(self):
        """Test that --color works as a flag without requiring value."""
        # When using --color without a value, it should default to rainbow
        mock_rainbow = unittest.mock.MagicMock(return_value="Rainbow text")
        
        with patch("figlet_forge.cli.main.get_coloring_functions", return_value=mock_rainbow):
            # Use --color as a flag (should default to rainbow)
            main(["--color", "Hello"])

            # The rainbow colorizer should have been called (applied to the rendered text)
            mock_rainbow.assert_called_once()

    def test_sample_text_without_value(self):
        """Test that --sample-text works without a value."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            main(["--sample", "--sample-text"])
            mock_showcase.assert_called_with(
                sample_text="Hello Eidos", fonts=None, color=None
            )

    def test_special_formatting_options(self):
        """Test special formatting options like equals sign."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            # Test special formatting with equals sign
            main(["--sample=Custom Text"])
            mock_showcase.assert_called_with(
                sample_text="Custom Text", fonts=None, color=None
            )

            # Reset mock
            mock_showcase.reset_mock()

            # Test combination of special formats
            main(["--showcase=Test", "--sample-color=RED"])
            mock_showcase.assert_called_with(
                sample_text="Test", fonts=None, color="RED"
            )

    def test_complex_option_combinations(self):
        """Test complex combinations of options."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            main(
                [
                    "--sample",
                    "--sample-text=Complex",
                    "--sample-color=ALL",
                    "--sample-fonts=ALL",
                ]
            )
            mock_showcase.assert_called_with(
                sample_text="Complex", fonts="ALL", color="ALL"
            )


if __name__ == "__main__":
    unittest.main()
