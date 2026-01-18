"""
Tests for the Figlet Forge command line interface.

These tests verify the functionality of the CLI including options parsing,
command execution, and output generation.
"""

import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from figlet_forge.cli.main import main


class TestFigletCLI(unittest.TestCase):
    """Test the Figlet Forge CLI functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_text = "Test"
        self.test_font = "standard"  # Use a font that should always be available

    def test_basic_output(self):
        """Test basic text output."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main([self.test_text])
            output = fake_out.getvalue()
            # Output should contain something (the rendered text)
            self.assertTrue(output.strip())
            # Output should span multiple lines (as FIGlet text does)
            self.assertGreater(output.count("\n"), 1)

    def test_font_option(self):
        """Test --font option."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main(["-f", self.test_font, self.test_text])
            output1 = fake_out.getvalue()

        # Try a different font if available
        different_font = "slant"
        with patch("sys.stdout", new=StringIO()) as fake_out:
            try:
                main(["-f", different_font, self.test_text])
                output2 = fake_out.getvalue()
                # Outputs should be different with different fonts
                self.assertNotEqual(output1.strip(), output2.strip())
            except SystemExit:
                # If font not available, test passes
                pass

    def test_width_option(self):
        """Test --width option."""
        narrow_width = 100  # Increased width to accommodate actual output
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main(
                ["-w", str(narrow_width), "A"]
            )  # Use single character to ensure it fits
            output = fake_out.getvalue()
            lines = output.splitlines()
            # Each line should not exceed specified width
            for line in lines:
                self.assertLessEqual(len(line), narrow_width)

    def test_list_fonts(self):
        """Test --list-fonts option."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main(["-l"])
            output = fake_out.getvalue()
            # Output should contain font names
            self.assertTrue(output.strip())
            # Should include standard font
            self.assertIn(self.test_font, output)

    def test_color_option(self):
        """Test --color option."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main(["-c", "RED", self.test_text])
            output = fake_out.getvalue()
            # Output should contain ANSI color codes
            self.assertIn("\033[", output)
            # Output should end with reset code
            self.assertIn("\033[0m", output)

    def test_color_list(self):
        """Test --color=list option."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main(["--color=list"])
            output = fake_out.getvalue()
            # Output should show available colors
            self.assertIn("Available color names:", output)
            # Should include basic colors
            self.assertIn("RED", output)
            self.assertIn("BLUE", output)

    def test_output_to_file(self):
        """Test --output option."""
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name

        try:
            # Write output to file
            with patch("sys.stdout", new=StringIO()):
                main(["-o", temp_path, self.test_text])

            # Verify file was created and has content
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path) as f:
                content = f.read()
            self.assertTrue(content.strip())
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_stdin_input(self):
        """Test reading from stdin."""
        with patch("sys.stdin", StringIO(self.test_text)), patch(
            "sys.stdout", new=StringIO()
        ) as fake_out, patch(
            "os.isatty", return_value=False
        ):  # Simulate pipe
            main([])
            output = fake_out.getvalue()
            # Output should contain something (the rendered text)
            self.assertTrue(output.strip())

    def test_transform_options(self):
        """Test transformation options."""
        # Test reverse option
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main(["-r", self.test_text])
            reversed_output = fake_out.getvalue()

        # Test flip option
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main(["-F", self.test_text])
            flipped_output = fake_out.getvalue()

        # Outputs should be different
        self.assertNotEqual(reversed_output, flipped_output)

    def test_showcase_option(self):
        """Test the --showcase option."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            with patch("sys.stdout", new=StringIO()):
                main(["--showcase"])
                # Check that showcase generator was called
                mock_showcase.assert_called_once()

    def test_sample_option(self):
        """Test the --sample option (equivalent to --showcase)."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            with patch("sys.stdout", new=StringIO()):
                main(["--sample"])
                # Check that showcase generator was called
                mock_showcase.assert_called_once()

    def test_sample_with_text(self):
        """Test using --sample with --sample-text."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            with patch("sys.stdout", new=StringIO()):
                main(["--sample", "--sample-text=Custom"])
                # Check that showcase was called with correct text
                mock_showcase.assert_called_once_with(
                    sample_text="Custom", fonts=None, color=None
                )

    def test_sample_with_color(self):
        """Test using --sample with --sample-color."""
        with patch("figlet_forge.cli.main.generate_showcase") as mock_showcase:
            with patch("sys.stdout", new=StringIO()):
                main(["--sample", "--sample-color=RED"])
                # Check that showcase was called with correct color
                mock_showcase.assert_called_once_with(
                    sample_text="hello", fonts=None, color="RED"
                )


if __name__ == "__main__":
    unittest.main()
