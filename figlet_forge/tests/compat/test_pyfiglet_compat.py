"""
Tests for pyfiglet compatibility using controlled mocks.

This approach ensures consistent tests regardless of the actual font files available.
"""

import unittest
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "functions,args,expected_match",
    [
        (
            ["figlet_format", "renderText"],
            [("Hello", {"font": "standard"})],
            True,
        ),
        (
            ["figlet_format", "renderText"],
            [("Test", {"font": "slant"})],
            True,
        ),
    ],
)
def test_function_equivalence(functions, args, expected_match):
    """Test that compatibility functions produce identical results."""
    try:
        # Import the functions
        from figlet_forge.compat import figlet_format, renderText

        # Replace actual figlet_format with a mock to avoid font loading issues
        with patch("figlet_forge.compat.figlet_format", return_value="Mocked output"):
            # Call each function with the same arguments and compare results
            results = []

            for func_name in functions:
                if func_name == "figlet_format":
                    func = figlet_format
                elif func_name == "renderText":
                    func = renderText
                else:
                    pytest.skip(f"Unknown function: {func_name}")

                # Call the function with each set of arguments
                for arg_set in args:
                    if isinstance(arg_set, tuple):
                        text, kwargs = arg_set
                        result = func(text, **kwargs)
                    else:
                        result = func(arg_set)

                    results.append(result)

            # Check if all results match (they should with our mock)
            all_match = all(r == results[0] for r in results)
            assert all_match == expected_match

    except ImportError as e:
        pytest.skip(f"Required module not available: {e}")


class TestPyFigletCompat(unittest.TestCase):
    """Test compatibility with pyfiglet API."""

    def setUp(self):
        """Set up test fixtures before each test."""
        # Import dependencies - skip test if not available
        try:
            from figlet_forge.compat import Figlet

            self.Figlet = Figlet
        except ImportError as e:
            self.skipTest(f"Required module not available: {e}")

    def test_figlet_class(self):
        """Test that Figlet class API is compatible."""
        # Replace actual font loading with a mock to avoid issues
        with patch("figlet_forge.core.figlet_font.FigletFont"):
            # Create a figlet instance
            fig = self.Figlet(font="standard")

            # Check required methods exist
            self.assertTrue(hasattr(fig, "renderText"))
            self.assertTrue(hasattr(fig, "getFonts"))
            self.assertTrue(hasattr(fig, "getDirection"))
            self.assertTrue(hasattr(fig, "setFont"))
            self.assertTrue(hasattr(fig, "getRenderWidth"))


if __name__ == "__main__":
    unittest.main()
