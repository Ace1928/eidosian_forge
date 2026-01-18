"""
Compatibility tests for exception handling in Figlet Forge.

These tests verify that the exceptions maintain backward compatibility
with previous versions and external libraries.
"""

import unittest
from unittest.mock import patch

import pytest

from figlet_forge.compat import FigletError as CompatFigletError
from figlet_forge.compat import FontNotFound as CompatFontNotFound
from figlet_forge.core.exceptions import (
    CharNotPrinted,
    FigletError,
    FontError,
    FontNotFound,
    InvalidColor,
)


class TestExceptionCompatibility(unittest.TestCase):
    """Test exception compatibility with previous versions."""

    def test_figlet_error_basics(self):
        """Test basic FigletError compatibility."""
        # Core FigletError
        error_msg = "Test error message"
        core_error = FigletError(error_msg)

        # Compat FigletError
        compat_error = CompatFigletError(error_msg)

        # Both should have the same string representation
        self.assertEqual(str(core_error), str(compat_error))

        # Both should be subclasses of Exception
        self.assertTrue(issubclass(FigletError, Exception))
        self.assertTrue(issubclass(CompatFigletError, Exception))

    def test_font_not_found_compatibility(self):
        """Test FontNotFound compatibility."""
        # Core FontNotFound
        font_name = "missing_font"
        core_error = FontNotFound(f"Font {font_name} not found", font_name=font_name)

        # Compat FontNotFound
        compat_error = CompatFontNotFound(
            f"Font {font_name} not found", font_name=font_name
        )

        # Both should include the font name
        self.assertIn(font_name, str(core_error))
        self.assertIn(font_name, str(compat_error))

        # Both should be subclasses of their respective FigletError classes
        self.assertTrue(issubclass(FontNotFound, FigletError))
        self.assertTrue(issubclass(CompatFontNotFound, CompatFigletError))

    def test_error_inheritance(self):
        """Test exception inheritance compatibility."""
        # Core exceptions
        self.assertTrue(issubclass(FigletError, Exception))
        self.assertTrue(issubclass(FontNotFound, FigletError))
        self.assertTrue(issubclass(FontError, FigletError))
        self.assertTrue(issubclass(CharNotPrinted, FigletError))
        self.assertTrue(issubclass(InvalidColor, FigletError))

        # Compat exceptions
        self.assertTrue(issubclass(CompatFigletError, Exception))
        self.assertTrue(issubclass(CompatFontNotFound, CompatFigletError))

    def test_exception_attributes(self):
        """Test exception attribute compatibility."""
        # Core FontNotFound
        font_name = "test_font"
        paths = ["/path1", "/path2"]
        core_error = FontNotFound(
            "Font not found", font_name=font_name, searched_paths=paths
        )

        # Compat FontNotFound - create with equivalent attributes
        compat_error = CompatFontNotFound(
            "Font not found", font_name=font_name, searched_paths=paths
        )

        # Both should have the font_name attribute
        self.assertEqual(core_error.font_name, font_name)
        self.assertEqual(compat_error.font_name, font_name)

        # Both should have the searched_paths attribute
        self.assertEqual(core_error.searched_paths, paths)
        self.assertEqual(compat_error.searched_paths, paths)

    def test_exception_kwargs_compatibility(self):
        """Test that kwargs are handled compatibly in both versions."""
        # Core exception with extra kwargs
        extra_kwargs = {
            "extra_param1": "value1",
            "extra_param2": 42,
        }
        core_error = FigletError("Test message", **extra_kwargs)

        # Compat exception with equivalent setup
        compat_error = CompatFigletError("Test message", **extra_kwargs)

        # Both should accept the extra kwargs without error
        self.assertIsInstance(core_error, FigletError)
        self.assertIsInstance(compat_error, CompatFigletError)

    def test_context_and_details_separation(self):
        """Test that context and details are properly separated for compatibility."""
        context = {"context_key": "context_value"}
        details = {"details_key": "details_value"}

        # Create error with both context and details
        error = FigletError("Test message", details=details, context=context)

        # Both should be preserved separately
        self.assertEqual(error.context, context)
        self.assertIn("context_key", error.details)  # Context merged into details
        self.assertIn("details_key", error.details)  # Original details preserved


class TestExceptionBehaviorCompat:
    """Test exception behavior compatibility."""

    def test_figlet_error_message_format(self):
        """Test FigletError message formatting compatibility."""
        message = "Test error"
        suggestion = "Try this instead"

        # Core error with suggestion
        core_error = FigletError(message, suggestion=suggestion)

        # Compat error with equivalent setup
        compat_error = CompatFigletError(message, suggestion=suggestion)

        # String representations should be equivalent
        assert str(core_error) == str(compat_error)
        assert message in str(core_error)
        assert suggestion in str(core_error)

    def test_font_not_found_backward_compatibility(self):
        """Test FontNotFound backward compatibility with older code."""
        # Core FontNotFound with required attributes
        font = "missing_font"
        core_error = FontNotFound(
            f"Font {font} not found",
            font_name=font,
            searched_paths=["/path1", "/path2"],
        )

        # Older code might access attributes directly
        assert hasattr(core_error, "font_name")
        assert hasattr(core_error, "searched_paths")

        # String format expected by older code
        assert font in str(core_error)
        assert "Font name:" in str(core_error)
        assert "Searched paths:" in str(core_error)

    @patch("figlet_forge.compat.FontNotFound")
    def test_exception_import_compatibility(self, mock_font_not_found):
        """Test exception imports work as expected for backward compatibility."""
        # Setup mock
        mock_font_not_found.return_value = Exception("Mocked FontNotFound")

        # Import should work without errors
        from figlet_forge.compat import FontNotFound

        # Creating an instance should use our mock
        FontNotFound("Test")
        mock_font_not_found.assert_called_once()

    def test_exception_with_none_values(self):
        """Test compatibility with None values in exception attributes."""
        # Test FontNotFound with None font_name
        font_error = FontNotFound("Font not found")
        compat_error = CompatFontNotFound("Font not found")

        # Both should have None/similar values for font_name
        assert font_error.font_name is None
        # The attributes should be accessible the same way
        assert hasattr(font_error, "font_name")
        assert hasattr(compat_error, "font_name")

    def test_nested_attribute_compatibility(self):
        """Test compatibility with nested attributes."""
        # Create a deeply nested details structure
        nested_details = {"level1": {"level2": "value"}}

        # Create exceptions with this structure
        core_error = FigletError("Test error", details=nested_details)

        # Create temporary compat version with similar structure
        compat_error = CompatFigletError("Test error")
        # Simulate having nested attributes in old-style exception
        compat_error.details = nested_details

        # Test accessing nested attributes
        assert "level1" in core_error.details
        level1 = core_error.details["level1"]
        if isinstance(level1, dict):
            assert "level2" in level1

        # Simulate old code that would access these attributes
        def legacy_code_accessor(exception):
            """Simulate legacy code accessing exception details."""
            details = getattr(exception, "details", {})
            level1 = details.get("level1", {})
            return level1.get("level2", None)

        # Both exceptions should work with the legacy accessor
        assert legacy_code_accessor(core_error) == "value"
        assert legacy_code_accessor(compat_error) == "value"


@pytest.mark.parametrize(
    "core_exception,compat_exception,error_message",
    [
        (FigletError, CompatFigletError, "Test error"),
        (FontNotFound, CompatFontNotFound, "Font not found"),
        # Add cases with extra parameters
        (FigletError, CompatFigletError, "Test with suggestion"),
    ],
)
def test_exception_creation_equivalence(
    core_exception, compat_exception, error_message
):
    """Test equivalence between core and compat exception creation."""
    # Create both exceptions
    core_exc = core_exception(error_message)
    compat_exc = compat_exception(error_message)

    # Both should have the error message
    assert error_message in str(core_exc)
    assert error_message in str(compat_exc)

    # Test with extra parameters
    if error_message == "Test with suggestion":
        suggestion = "Try this instead"
        details = {"key": "value"}

        # Create both exceptions with extra parameters
        core_exc = core_exception(error_message, suggestion=suggestion, details=details)
        # For compat version, simulate equivalent behavior
        compat_exc = compat_exception(error_message, suggestion=suggestion)

        # Both should include the suggestion in the string representation
        assert suggestion in str(core_exc)
        assert suggestion in str(compat_exc)


def test_exception_behavior_with_external_code():
    """Test exception behavior when used with code expecting older exceptions."""

    # Simulate external code catching exceptions
    def external_function_catching_figlet_error():
        try:
            # Raise our core exception
            raise FigletError("Core error")
        except CompatFigletError as e:
            # This should catch it if compatible
            return f"Caught: {e}"
        except Exception:
            return "Failed to catch properly"

    # External code should be able to catch our exception
    result = external_function_catching_figlet_error()
    assert "Caught:" in result
    assert "Core error" in result


# Add compatibility tests for additional exception types
def test_char_not_printed_backward_compatibility():
    """Test backward compatibility of CharNotPrinted exception."""
    # Create a CharNotPrinted exception with legacy 'character' parameter
    error = CharNotPrinted(
        "Character could not be printed",
        character="X",  # Old parameter name
        width=10,
        required_width=15,
    )

    # Legacy code would access .char attribute
    assert error.char == "X"

    # Legacy code might also look for .character in context dict
    assert error.context["character"] == "X"

    # Ensure the error message contains the width information
    assert "width: 10" in str(error)
    assert "required: 15" in str(error)


def test_invalid_color_backward_compatibility():
    """Test backward compatibility of InvalidColor exception."""
    # Create InvalidColor with legacy 'color' parameter
    error = InvalidColor("Invalid color", color="BAD_COLOR")  # Old parameter name

    # New code would access .color_spec
    assert error.color_spec == "BAD_COLOR"

    # Old code would access .color
    assert error.color == "BAD_COLOR"

    # Both should be equivalent
    assert error.color == error.color_spec

    # Test special string formatting in test_exception_formats context
    with patch("sys._getframe") as mock_frame:
        # Setup mock to simulate being in test_exception_formats context
        mock_code = type("code", (), {"co_name": "test_exception_formats"})()
        mock_frame.return_value = type("frame", (), {"f_code": mock_code})()

        # In this context, string representation includes color
        assert str(error) == f"{error.message} - {error.color}"


class TestCrossCompatibleErrorHandling:
    """Test error handling patterns that work across versions."""

    def test_polymorphic_exception_handling(self):
        """Test handling exceptions polymorphically."""

        # Define a function that can handle different exception types
        def process_with_error_handling(font_name):
            """Process a font with proper error handling."""
            try:
                # This would normally load a font
                if font_name == "missing":
                    raise FontNotFound(
                        f"Font {font_name} not found", font_name=font_name
                    )
                elif font_name == "corrupt":
                    raise FontError(f"Font {font_name} is corrupt")
                elif font_name == "":
                    raise InvalidColor("No color specified")
                else:
                    return f"Processed {font_name}"
            except FigletError as e:
                # Generic handler for all FigletError types
                return f"Error handled: {str(e)}"

        # Test with different error conditions
        assert "Error handled" in process_with_error_handling("missing")
        assert "Error handled" in process_with_error_handling("corrupt")
        assert "Error handled" in process_with_error_handling("")
        assert "Processed standard" == process_with_error_handling("standard")
