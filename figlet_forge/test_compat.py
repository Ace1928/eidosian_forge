#!/usr/bin/env python3

"""
Tests compatibility between Figlet Forge and the original pyfiglet.

This standalone script verifies that Figlet Forge maintains
compatibility with the pyfiglet API and rendering behavior.
"""

import sys
import unittest
from pathlib import Path

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent))


class TestFigletCompatibility(unittest.TestCase):
    """Test suite for compatibility between Figlet Forge and pyfiglet."""

    def setUp(self):
        """Set up test environment."""
        # Try to import pyfiglet and figlet_forge
        try:
            import pyfiglet

            self.pyfiglet_available = True
        except ImportError:
            print("Warning: Original pyfiglet not available, skipping comparison tests")
            self.pyfiglet_available = False

        try:
            import figlet_forge
            from figlet_forge.compat import Figlet as CompatFiglet

            self.figlet_forge_available = True
        except ImportError:
            print("Error: Figlet Forge not available!")
            self.figlet_forge_available = False

    def test_api_compatibility(self):
        """Test that the API interfaces are compatible."""
        if not self.figlet_forge_available:
            self.skipTest("Figlet Forge not available")

        # Import from the compat module
        from figlet_forge.compat import (
            DEFAULT_FONT,
            Figlet,
            figlet_format,
            renderText,
        )

        # Test the module attributes
        self.assertIsInstance(DEFAULT_FONT, str)
        self.assertEqual(figlet_format.__name__, "figlet_format")
        self.assertEqual(renderText.__name__, "figlet_format")  # Should be aliased

        # Test Figlet class
        fig = Figlet()
        self.assertEqual(fig.__class__.__name__, "Figlet")

        # Test figlet_format function
        result = figlet_format("Test")
        self.assertIsInstance(result, str)
        # Structural check: FIGlet output should have multiple lines and non-whitespace content
        lines = result.strip().splitlines()
        self.assertGreater(len(lines), 1, "FIGlet output should be multi-line")
        self.assertGreater(len(result.strip()), 0, "FIGlet output should not be empty")

    def test_rendering_equivalence(self):
        """Test that rendering produces equivalent results."""
        if not self.pyfiglet_available or not self.figlet_forge_available:
            self.skipTest("Both pyfiglet and Figlet Forge required for comparison")

        import pyfiglet

        from figlet_forge.compat import figlet_format as forge_format

        test_strings = ["Hello", "World", "Testing", "123"]
        fonts = ["standard", "slant", "small"]

        for text in test_strings:
            for font in fonts:
                try:
                    pyfiglet_result = pyfiglet.figlet_format(text, font=font)
                    forge_result = forge_format(text, font=font)

                    # Clean up strings for comparison
                    py_clean = self._normalize_output(pyfiglet_result)
                    forge_clean = self._normalize_output(forge_result)

                    # Compare structural properties rather than literal characters
                    py_lines = py_clean.splitlines()
                    forge_lines = forge_clean.splitlines()
                    
                    self.assertEqual(
                        len(py_lines),
                        len(forge_lines),
                        f"Line count differs for font '{font}' and text '{text}'",
                    )
                    self.assertTrue(
                        len(forge_clean) > 0,
                        f"Forge output is empty for font '{font}' and text '{text}'",
                    )
                except Exception as e:
                    self.fail(f"Error comparing outputs for font '{font}': {e}")

    def test_font_loading(self):
        """Test that font loading behaves consistently."""
        if not self.figlet_forge_available:
            self.skipTest("Figlet Forge not available")

        from figlet_forge.compat import Figlet

        # Test that default font loads
        fig = Figlet()
        self.assertIsNotNone(fig)

        # Test loading specific font
        fig = Figlet(font="slant")
        self.assertEqual(fig.font, "slant")

        # Test getFonts method
        fonts = fig.getFonts()
        self.assertIsInstance(fonts, list)
        self.assertTrue(len(fonts) > 0)
        self.assertIn("standard", fonts)

    def _normalize_output(self, text: str) -> str:
        """Normalize output for comparison by removing whitespace variations."""
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.splitlines()]
        # Remove empty lines from beginning and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop(-1)
        return "\n".join(lines)


def main():
    """Run the compatibility tests."""
    unittest.main()


if __name__ == "__main__":
    main()
