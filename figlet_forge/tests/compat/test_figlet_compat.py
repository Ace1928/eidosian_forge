"""
Unit tests for Figlet compatibility module.

This module tests the compatibility layer that ensures Figlet Forge can
be used as a drop-in replacement for pyfiglet.
"""

import unittest

from figlet_forge.compat.figlet_compat import Figlet, figlet_format, renderText
from figlet_forge.core.exceptions import FontNotFound


class TestFigletCompat(unittest.TestCase):
    """Test the figlet_compat module."""

    def test_figlet_class_initialization(self):
        """Test that Figlet class initializes correctly."""
        fig = Figlet()
        self.assertEqual(fig.font, "standard")
        self.assertEqual(fig.direction, "auto")
        self.assertEqual(fig.justify, "auto")
        self.assertEqual(fig.width, 80)

    def test_render_text(self):
        """Test rendering text with defaults."""
        fig = Figlet()
        result = fig.renderText("Test")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_get_fonts(self):
        """Test getting available fonts."""
        fig = Figlet()
        fonts = fig.getFonts()
        self.assertIsInstance(fonts, list)
        self.assertIn("standard", fonts)

    def test_set_font(self):
        """Test setting the font."""
        fig = Figlet()
        fig.setFont("slant")
        self.assertEqual(fig.font, "slant")
        result = fig.renderText("Test")
        self.assertIsInstance(result, str)

    def test_get_direction(self):
        """Test getting the direction."""
        fig = Figlet(direction="right-to-left")
        direction = fig.getDirection()
        self.assertEqual(direction, "right_to_left")  # Note the underscore format

    def test_get_justify(self):
        """Test getting the justification."""
        fig = Figlet(justify="center")
        justify = fig.getJustify()
        self.assertEqual(justify, "center")

    def test_get_render_width(self):
        """Test getting the render width."""
        fig = Figlet()
        width = fig.getRenderWidth("Test")
        self.assertIsInstance(width, int)
        self.assertGreater(width, 0)

    def test_figlet_format(self):
        """Test figlet_format function."""
        result = figlet_format("Test")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_render_text_alias(self):
        """Test renderText function (alias for figlet_format)."""
        result = renderText("Test")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Verify they produce the same output
        result2 = figlet_format("Test")
        self.assertEqual(result, result2)

    def test_font_not_found(self):
        """Test handling of non-existent fonts."""
        with self.assertRaises(FontNotFound):
            fig = Figlet(font="nonexistent_font")
            # Force font loading and error
            fig.renderText("Test")

    def test_direction_conversion(self):
        """Test direction format conversion."""
        fig = Figlet()

        # Test each direction format
        self.assertEqual(fig._convert_direction("auto"), "auto")
        self.assertEqual(fig._convert_direction("left-to-right"), "left-to-right")
        self.assertEqual(fig._convert_direction("left_to_right"), "left-to-right")
        self.assertEqual(fig._convert_direction("ltr"), "left-to-right")
        self.assertEqual(fig._convert_direction("right-to-left"), "right-to-left")
        self.assertEqual(fig._convert_direction("right_to_left"), "right-to-left")
        self.assertEqual(fig._convert_direction("rtl"), "right-to-left")

        # Test unknown direction
        self.assertEqual(fig._convert_direction("unknown"), "auto")


if __name__ == "__main__":
    unittest.main()
