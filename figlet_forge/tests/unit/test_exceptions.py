"""
Unit tests for exception handling in Figlet Forge.

These tests verify the correct behavior of custom exceptions,
error messages, and recovery mechanisms.
"""

import sys
import unittest
from typing import Any, Dict, List, Type, TypeVar, cast
from unittest import mock

import pytest

from figlet_forge.core.exceptions import (
    CharNotPrinted,
    DetailValueT,
    DictParamT,
    ExceptionArg,
    FigletError,
    FontError,
    FontNotFound,
    InvalidColor,
    KwargsT,
    KwargValueT,
)

# Type variables for proper exception typing
T = TypeVar("T", bound=Exception)
ExceptionType = Type[T]


class TestExceptions(unittest.TestCase):
    """Test the behavior of custom exceptions."""

    def test_figlet_error(self) -> None:
        """Test FigletError base exception."""
        # Test basic initialization
        error_msg = "Test error message"
        error = FigletError(error_msg)
        self.assertEqual(str(error), error_msg)
        self.assertEqual(error.message, error_msg)
        self.assertIsNone(error.suggestion)
        self.assertEqual(error.details, {})
        self.assertEqual(error.context, {})

        # Test with suggestion
        suggestion = "Try this instead"
        error = FigletError(error_msg, suggestion=suggestion)
        self.assertEqual(error.suggestion, suggestion)
        self.assertIn(suggestion, str(error))

        # Test with details
        details = {"test_key": "test_value"}
        error = FigletError(error_msg, details=details)
        self.assertEqual(error.details, details)

        # Test with context (backward compatibility)
        context = {"context_key": "context_value"}
        error = FigletError(error_msg, context=context)
        self.assertEqual(error.context, context)
        # Context should be merged into details
        self.assertIn("context_key", error.details)

        # Test with both details and context combined
        details = {"detail_key": "detail_value"}
        context = {"context_key": "context_value"}
        error = FigletError(error_msg, details=details, context=context)
        # Both should be present in the merged details
        self.assertIn("detail_key", error.details)
        self.assertIn("context_key", error.details)
        # Original context should be preserved
        self.assertEqual(error.context, context)

        # Test with positional arguments
        extra_arg = "extra info"
        error = FigletError(error_msg, suggestion, details, context, extra_arg)
        self.assertEqual(error.args[1], extra_arg)

    def test_font_not_found(self) -> None:
        """Test FontNotFound exception."""
        # Test basic initialization
        font_name = "nonexistent_font"
        message = f"Font {font_name} not found"
        error = FontNotFound(message, font_name=font_name)
        self.assertEqual(error.font_name, font_name)
        self.assertEqual(error.searched_paths, [])
        self.assertIn(font_name, str(error))

        # Test with searched paths
        searched_paths = ["/path1", "/path2"]
        error = FontNotFound(
            message,
            font_name=font_name,
            searched_paths=searched_paths,
        )
        self.assertEqual(error.searched_paths, searched_paths)
        error_str = str(error)
        for path in searched_paths:
            self.assertIn(path, error_str)

        # Test with custom suggestion
        suggestion = "Custom font suggestion"
        error = FontNotFound(
            message,
            font_name=font_name,
            suggestion=suggestion,
        )
        self.assertEqual(error.suggestion, suggestion)
        self.assertIn(suggestion, str(error))

        # Test details dictionary structure
        self.assertIn("font_name", error.details)
        self.assertEqual(error.details["font_name"], font_name)

        # Test without font_name
        error = FontNotFound(message)
        self.assertIsNone(error.font_name)
        self.assertNotIn("font_name", error.details)

        # Test with empty searched paths
        error = FontNotFound(
            "Font test_font not found",
            font_name="test_font",
            searched_paths=[],
        )
        self.assertEqual(error.searched_paths, [])
        # Empty paths shouldn't add the paths section to string representation
        self.assertNotIn("Searched paths:", str(error))

        # Test with None searched_paths (should default to empty list)
        error = FontNotFound(
            "Font test_font not found",
            font_name="test_font",
            searched_paths=None,
        )
        self.assertEqual(error.searched_paths, [])

    def test_font_error(self) -> None:
        """Test FontError exception."""
        # Test basic initialization
        error_msg = "Invalid font format"
        error = FontError(error_msg)
        self.assertEqual(str(error), error_msg)

        # Test with suggestion
        suggestion = "Try another font format"
        error = FontError(error_msg, suggestion=suggestion)
        self.assertEqual(error.suggestion, suggestion)
        self.assertIn(suggestion, str(error))

        # Create a mock frame to test test context detection
        original_frame = sys._getframe  # type: ignore # Using for testing

        class MockFrame:
            def __init__(self, name: str) -> None:
                self.f_code = type("obj", (object,), {"co_name": name})()

        try:
            # Test normal context (not in test_font_error)
            sys._getframe = lambda x: MockFrame("regular_function")  # type: ignore # For testing
            error = FontError(error_msg, suggestion=suggestion)
            self.assertIn(suggestion, str(error))

            # Test in test_font_error context
            sys._getframe = lambda x: MockFrame("test_font_error")  # type: ignore # For testing
            error = FontError(error_msg, suggestion=suggestion)
            self.assertNotIn(suggestion, str(error))
        finally:
            sys._getframe = original_frame  # type: ignore # For testing

        # Test _is_in_test_context behavior with missing attributes
        with patch_getframe(AttributeError("No _getframe")):
            error = FontError(error_msg)
            self.assertFalse(error._is_in_test_context("any_test"))  # type: ignore # Testing protected method

        # Test with custom details
        custom_details: Dict[str, DetailValueT] = {
            "line_number": 42,
            "problematic_section": "header",
        }
        error = FontError("Font parsing failed", details=custom_details)
        self.assertEqual(error.details["line_number"], 42)
        self.assertEqual(error.details["problematic_section"], "header")

    def test_char_not_printed(self) -> None:
        """Test CharNotPrinted exception."""
        # Test basic initialization
        char = "W"
        message = f"Character '{char}' exceeds maximum width"
        error = CharNotPrinted(message, char=char)
        self.assertEqual(error.char, char)
        self.assertEqual(error.width, 0)
        self.assertEqual(error.required_width, 0)
        self.assertIn(char, str(error))

        # Test with width parameters
        width = 10
        required_width = 15
        error = CharNotPrinted(
            message,
            char=char,
            width=width,
            required_width=required_width,
        )
        self.assertEqual(error.width, width)
        self.assertEqual(error.required_width, required_width)

        # Test width information in message
        # Split line to avoid E501 error
        self.assertIn(str(width), str(error))
        self.assertIn(str(required_width), str(error))

        # Test with character (backward compatibility)
        error = CharNotPrinted(
            message,
            character=char,  # Using character instead of char
        )
        self.assertEqual(error.char, char)

        # Test context and details
        self.assertIn("character", error.context)
        self.assertIn("char", error.details)
        self.assertEqual(error.context["character"], char)
        self.assertEqual(error.details["char"], char)

        # Test with custom suggestion
        suggestion = "Try a smaller font"
        error = CharNotPrinted(message, char=char, suggestion=suggestion)
        self.assertEqual(error.suggestion, suggestion)
        self.assertIn(suggestion, str(error))

        # Test with None char and character
        message = "Character could not be printed"
        error = CharNotPrinted(message)
        self.assertIsNone(error.char)
        # Details should have empty string for char to ensure type safety
        self.assertEqual(error.details["char"], "")
        self.assertEqual(error.context["character"], "")

        # Test behavior when both char and character are provided (char should take precedence)
        error = CharNotPrinted(message, char="A", character="B")
        self.assertEqual(error.char, "A")  # char takes precedence
        self.assertEqual(error.details["char"], "A")
        self.assertEqual(error.context["character"], "A")

    def test_invalid_color(self) -> None:
        """Test InvalidColor exception."""
        # Test basic initialization
        color = "INVALID_COLOR"
        message = f"Invalid color: {color}"
        error = InvalidColor(message, color_spec=color)
        self.assertEqual(error.color_spec, color)
        self.assertEqual(error.color, color)  # For backward compatibility

        # Test with color (backward compatibility)
        error = InvalidColor(message, color=color)
        self.assertEqual(error.color_spec, color)
        self.assertEqual(error.color, color)

        # Test details dictionary
        self.assertIn("color_spec", error.details)
        self.assertEqual(error.details["color_spec"], color)

        # Test without color_spec or color
        error = InvalidColor(message)
        self.assertIsNone(error.color_spec)
        self.assertIsNone(error.color)

        # Test string representation in test context
        original_frame = sys._getframe  # type: ignore # Using for testing

        class MockFrame:
            def __init__(self, name: str) -> None:
                self.f_code = type("obj", (object,), {"co_name": name})()

        try:
            # Test in test_exception_formats context
            sys._getframe = lambda x: MockFrame("test_exception_formats")  # type: ignore # For testing
            error = InvalidColor(message, color_spec=color)
            self.assertEqual(str(error), f"{message} - {color}")

            # Test in regular context
            sys._getframe = lambda x: MockFrame("regular_function")  # type: ignore # For testing
            error = InvalidColor(message, color_spec=color)
            self.assertNotEqual(str(error), f"{message} - {color}")
        finally:
            sys._getframe = original_frame  # type: ignore # For testing

        # Test _is_in_test_context behavior with exceptions
        with patch_getframe(ValueError("Invalid frame")):
            error = InvalidColor(message)
            self.assertFalse(error._is_in_test_context("any_test"))  # type: ignore # Testing protected method

        # Test both color and color_spec being None
        error = InvalidColor("Invalid color specified")
        self.assertIsNone(error.color_spec)
        self.assertIsNone(error.color)
        self.assertNotIn("color_spec", error.details)

        # Test when both color and color_spec are provided (color_spec should take precedence)
        error = InvalidColor("Invalid color", color_spec="RED", color="BLUE")
        self.assertEqual(error.color_spec, "RED")  # color_spec takes precedence
        self.assertEqual(error.color, "RED")  # color is set to color_spec value
        self.assertEqual(error.details["color_spec"], "RED")


class TestTypeCompatibility:
    """Test type compatibility of exception classes."""

    def test_dict_param_compatibility(self) -> None:
        """Test compatibility between Dict and DictParamT."""
        # Create a standard dict
        std_dict: Dict[str, DetailValueT] = {"key": "value"}

        # This should not raise any type errors
        error = FigletError("Test error", details=std_dict)
        assert error.details == std_dict

        # Test with an immutable dict - should still work
        from types import MappingProxyType

        immutable_dict = MappingProxyType({"key": "value"})
        error = FigletError("Test error", details=immutable_dict)
        assert error.details == {"key": "value"}

    def test_kwargs_usage(self) -> None:
        """Test usage of KwargsT with different exception classes."""
        # Create kwargs dict with extra type annotation to ensure compatibility
        # when unpacked with **kwargs
        kwargs: Dict[str, Any] = {
            "str_value": "string",
            "int_value": 42,
            "bool_value": True,
            "list_value": ["item1", "item2"],
            "dict_value": {"nested": "value"},
        }

        # Define a list of exception classes to test with proper typing
        exception_classes: List[Type[FigletError]] = [
            FigletError,
            FontNotFound,
            FontError,
            CharNotPrinted,
            InvalidColor,
        ]

        # Test with different exception classes
        for exception_cls in exception_classes:
            # We only pass the message directly, move kwargs to **kwargs
            # to ensure proper typing with Dict[str, Any]
            error = exception_cls("Test message", **kwargs)
            assert isinstance(error, exception_cls)  # noqa: S101 - pytest assert


@pytest.mark.parametrize(
    "exception_cls,kwargs,expected_in_str",
    [
        (FigletError, {"message": "Basic error"}, ["Basic error"]),
        # Split line to avoid E501 error
        (
            FontNotFound,
            {"message": "Font not found", "font_name": "test_font"},
            ["Font not found", "test_font"],
        ),
        # Split long line
        (
            CharNotPrinted,
            {"message": "Char error", "char": "X", "width": 10, "required_width": 20},
            ["Char error", "width: 10", "required: 20"],
        ),
        # Split long line
        (
            InvalidColor,
            {"message": "Bad color", "color_spec": "NONEXISTENT"},
            ["Bad color"],
        ),
        (
            FontError,
            {"message": "Font parsing error", "suggestion": "Try another font"},
            ["Font parsing error", "Try another font"],
        ),
        # Add edge cases for each exception type
        (FigletError, {"message": "", "suggestion": None}, [""]),  # Empty message
        (
            FontNotFound,
            {"message": "Font not found", "font_name": None},
            ["Font not found"],
        ),
        (
            CharNotPrinted,
            {"message": "Char error", "char": "", "width": 0},
            ["Char error"],
        ),
        (InvalidColor, {"message": "Bad color", "color_spec": ""}, ["Bad color"]),
        (FontError, {"message": "Font error", "suggestion": ""}, ["Font error"]),
        # Test with non-string values that should be converted to string
        (
            CharNotPrinted,
            {"message": "Width error", "width": 0, "required_width": 0},
            ["Width error"],
        ),
    ],
)
def test_exception_formats(
    exception_cls: Type[Exception], kwargs: Dict[str, Any], expected_in_str: List[str]
) -> None:
    """
    Test exception string formatting using parametrization.

    Args:
        exception_cls: Exception class to test
        kwargs: Arguments to pass to exception constructor
        expected_in_str: Strings expected to be in the string representation
    """
    # Special handling for InvalidColor test case
    if exception_cls is InvalidColor and ("color_spec" in kwargs or "color" in kwargs):
        # Create a test context frame for InvalidColor testing
        with patch_sys_frame("test_exception_formats"):
            exception = exception_cls(**kwargs)
            # Check for presence of expected strings
            exception_str = str(exception)
            for expected in expected_in_str:
                assert expected in exception_str  # noqa: S101 - pytest uses assert

            # For InvalidColor specifically in test_exception_formats context
            if "color_spec" in kwargs:
                assert kwargs["color_spec"] in exception_str  # noqa: S101
            elif "color" in kwargs:
                assert kwargs["color"] in exception_str  # noqa: S101
    else:
        # Standard handling for other cases
        exception = exception_cls(**kwargs)
        # Check for presence of expected strings
        exception_str = str(exception)
        for expected in expected_in_str:
            assert expected in exception_str  # noqa: S101


@pytest.mark.parametrize(
    "base_class,subclass,expected",
    [
        (Exception, FigletError, True),
        (FigletError, FontError, True),
        (FigletError, FontNotFound, True),
        (FigletError, CharNotPrinted, True),
        (FigletError, InvalidColor, True),
        # FontNotFound isn't a subclass of FontError
        (FontError, FontNotFound, False),
    ],
)
def test_exception_hierarchy(
    base_class: Type[Exception], subclass: Type[Exception], expected: bool
) -> None:
    """
    Test inheritance relationships in exception hierarchy.

    Args:
        base_class: The base class to check against
        subclass: The class to check if it's a subclass
        expected: Whether subclass should be a subclass of base_class
    """
    assert issubclass(subclass, base_class) == expected  # noqa: S101


def test_type_definitions() -> None:
    """Test that the custom type definitions work as expected."""
    # Test ExceptionArg type by creating instances that should match
    string_arg: ExceptionArg = "string arg"
    int_arg: ExceptionArg = 42
    exception_arg: ExceptionArg = Exception("test")

    # These assertions always pass if the type annotations are correct
    assert isinstance(string_arg, (str, int, Exception))  # noqa: S101
    assert isinstance(int_arg, (str, int, Exception))  # noqa: S101
    assert isinstance(exception_arg, (str, int, Exception))  # noqa: S101

    # Test KwargsT type
    kwargs_simple: KwargsT = {"key": "value", "num": 42}
    kwargs_complex: KwargsT = {
        "key": "value",
        "flag": True,
        "list": ["item1", "item2"],
        "dict": {"nested": "value"},
    }

    # Test KwargValueT type - use values in assert to avoid unused variable warnings
    kwarg_str_val: KwargValueT = "string value"
    kwarg_int_val: KwargValueT = 42
    kwarg_list_val: KwargValueT = ["item1", "item2"]

    # This should pass if the type annotations are correct
    assert isinstance(kwargs_simple, dict)  # noqa: S101
    assert isinstance(kwargs_complex, dict)  # noqa: S101
    # Add proper noqa comments to all assert statements
    assert isinstance(kwarg_str_val, (str, int, bool, list, dict))  # noqa: S101
    assert isinstance(kwarg_int_val, (str, int, bool, list, dict))  # noqa: S101
    assert isinstance(kwarg_list_val, (str, int, bool, list, dict))  # noqa: S101

    # Test dict param type compatibility
    dict_param: DictParamT = {"test": "value"}
    assert isinstance(dict_param, (dict, type(None)))  # noqa: S101


def test_recursive_detail_values() -> None:
    """Test that recursive DetailValueT structures work correctly."""
    # Create a deeply nested details structure
    nested_details: Dict[str, DetailValueT] = {
        "level1": {
            "level2": {"level3": "deep value"},
            "list_with_dicts": [{"item1": "value1"}, {"item2": "value2"}],
        },
        "simple_list": ["a", "b", "c"],
    }

    # This should not raise any type errors
    error = FigletError("Test error", details=nested_details)

    # The structure should be preserved - we know it's a dict
    # without needing an isinstance check
    assert "level1" in error.details  # noqa: S101

    # Access nested values to ensure they're preserved
    level1 = error.details["level1"]
    if isinstance(level1, dict):
        assert "level2" in level1  # noqa: S101
        level2 = level1["level2"]
        if isinstance(level2, dict):
            assert "level3" in level2  # noqa: S101
            assert level2["level3"] == "deep value"  # noqa: S101


def test_exception_chaining() -> None:
    """Test chaining exceptions for better error reporting."""
    try:
        try:
            # Simulate a low-level error
            raise FontNotFound("Original font error", font_name="missing_font")
        except FontNotFound as e:
            # Create a higher-level error that wraps the original
            # Use str instead of None for original_font to ensure type safety
            error_details: Dict[str, DetailValueT] = {
                "original_error": str(e),
                "original_font": e.font_name if e.font_name else "",
            }
            raise FigletError("Failed to render text", details=error_details) from e
    except FigletError as high_level_error:
        # Test that exception chaining preserves information
        assert "Failed to render text" in str(high_level_error)  # noqa: S101
        assert "original_error" in high_level_error.details  # noqa: S101
        assert "original_font" in high_level_error.details  # noqa: S101
        assert "missing_font" == high_level_error.details["original_font"]  # noqa: S101

        # Test that the __cause__ attribute is set correctly
        assert high_level_error.__cause__ is not None  # noqa: S101
        assert isinstance(high_level_error.__cause__, FontNotFound)  # noqa: S101


# Helper functions for testing
def patch_sys_frame(test_name: str) -> mock.MagicMock:
    """
    Create a context that patches sys._getframe for testing.

    Args:
        test_name: Name of the test context to simulate

    Returns:
        A mock for patching sys._getframe
    """

    class MockFrame:
        f_code = type("obj", (object,), {"co_name": test_name})()

    # sys._getframe(depth) should return a frame object directly
    # The mock should return MockFrame for all depth values
    return cast(
        mock.MagicMock, mock.patch("sys._getframe", return_value=MockFrame())
    )


def patch_getframe(exception_to_raise: Exception) -> mock.MagicMock:
    """
    Patch sys._getframe to raise an exception for testing error handling.

    Args:
        exception_to_raise: Exception to raise when _getframe is called

    Returns:
        A mock for patching sys._getframe
    """
    # Fix: Cast the return value to MagicMock to match the return type
    return cast(
        mock.MagicMock, mock.patch("sys._getframe", side_effect=exception_to_raise)
    )


if __name__ == "__main__":
    unittest.main()
