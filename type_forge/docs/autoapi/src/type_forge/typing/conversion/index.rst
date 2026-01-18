.. py:module:: src.type_forge.typing.conversion

Type Conversion for the Type Forge system.

   This module provides robust type conversion utilities with comprehensive error handling,
   focusing on safety, composability, and explicit failure modes. The core components
   include the ConversionResult monad for chainable operations and specialized conversion
   functions for common Python types.

   All functions guarantee exception safety through explicit error handling patterns.


Module Contents
---------------


   .. py:class:: ConversionResult(success, value = None, error = None)   :module: 

      Result of a type conversion operation with success status and error tracking.

      This class encapsulates the result of a type conversion operation,
      providing both the converted value (if successful) and error information
      when conversion fails. It allows for chainable operations and error handling.

      .. rubric:: Attributes

      success (bool): Whether the conversion was successful
      value (Optional[T]): The converted value, or None if conversion failed
      error (Optional[str]): Error message if conversion failed

      .. rubric:: Examples

      >>> result = ConversionResult.create_success(42)
      >>> result.success
      True
      >>> print(result.value)
      42
      >>> result = ConversionResult[int].failure("Invalid conversion")
      >>> result.success
      False
      >>> print(result.error)
      Invalid conversion
      >>> # Chain operations
      >>> result = ConversionResult.create_success("123")
      >>> result.then(lambda s: ConversionResult.create_success(int(s))).value
      123

      Initialize a ConversionResult.

      :param success: Whether the conversion was successful
      :param value: The converted value, None if conversion failed
      :param error: Error message if conversion failed

      .. rubric:: Examples

      >>> result = ConversionResult(True, 42)
      >>> result.success
      True
      >>> result.value
      42
      >>> failed = ConversionResult(False, None, "Conversion error")
      >>> failed.error
      'Conversion error'



.. py:function:: coerce_to_type(value, target_type)

      Coerce a value to a target type, raising TypeError if conversion fails.

      Unlike try_convert, this function raises an exception if the value cannot
      be converted, making it suitable for validation scenarios where conversion
      failure should stop execution.

      :param value: The value to convert
      :param target_type: The type to convert to

      :returns: The value converted to the target type
      :rtype: T

      :raises TypeError: If the value cannot be converted to the target type

      .. rubric:: Examples

      >>> coerce_to_type("42", int)
      42
      >>> coerce_to_type(42, str)
      '42'
      >>> coerce_to_type(True, int)
      1
      >>> try:
      ...     coerce_to_type("not_a_number", int)
      ... except TypeError:
      ...     print("Conversion failed")
      Conversion failed

      .. note::

         This function is more strict than convert_with_fallback, raising an exception
         rather than returning the original value on failure.


.. py:function:: convert_with_fallback(value, primary_type, fallback_type)

      Try to convert a value to a primary type, with fallback to a secondary type.

      Attempts to convert the value to the primary type first, and if that fails,
      tries converting to the fallback type. If both fail, returns the original value.

      :param value: The value to convert
      :param primary_type: The preferred target type
      :param fallback_type: The fallback target type

      :returns: Converted value (T or R) or original value (S) if conversion fails
      :rtype: Union[T, R, S]

      .. rubric:: Examples

      >>> convert_with_fallback("123", int, float)
      123
      >>> convert_with_fallback("3.14", int, float)
      3.14
      >>> convert_with_fallback("hello", int, float)
      'hello'
      >>> convert_with_fallback(None, int, str)
      ''

      .. note::

         This function silently handles conversion errors and returns the original
         value if both conversions fail.


.. py:function:: register_converter(target_type, converter)

      Register a custom type converter for use with try_convert.

      :param target_type: The type to convert to
      :param converter: Function that attempts to convert an object to target_type

      .. rubric:: Examples

      >>> class CustomType:
      ...     def __init__(self, value: int):
      ...         self.value = value
      ...     def __eq__(self, other):
      ...         return isinstance(other, CustomType) and self.value == other.value
      >>> def custom_converter(value: object) -> ConversionResult[CustomType]:
      ...     try:
      ...         if isinstance(value, int):
      ...             return ConversionResult.create_success(CustomType(value))
      ...         elif isinstance(value, str) and value.isdigit():
      ...             return ConversionResult.create_success(CustomType(int(value)))
      ...         return ConversionResult.failure(
      ...             f"Cannot convert {type(value).__name__} to CustomType")
      ...     except Exception as e:
      ...         return ConversionResult.failure(str(e))
      >>> register_converter(CustomType, custom_converter)
      >>> result = try_convert(42, CustomType)
      >>> result.success
      True
      >>> result.value == CustomType(42)
      True


.. py:function:: safe_bool_convert(value)

      Safely convert a value to bool with semantic interpretation.

      Performs intelligent boolean conversion with special handling for
      string values like "yes", "no", "true", "false", etc.

      :param value: Any value that might be convertible to bool.

      :returns:

                A boolean representation of the value, with common string patterns
                      like "yes"/"no" properly handled.
      :rtype: bool

      .. rubric:: Examples

      >>> safe_bool_convert(True)
      True
      >>> safe_bool_convert(1)
      True
      >>> safe_bool_convert("yes")
      True
      >>> safe_bool_convert("false")
      False
      >>> safe_bool_convert(0)
      False
      >>> safe_bool_convert([])
      False
      >>> safe_bool_convert([1, 2, 3])
      True
      >>> safe_bool_convert(None)
      False

      .. note::

         This function interprets strings like "yes", "true", "1", "y", "t", "on" as True,
         and "no", "false", "0", "n", "f", "off" as False.


.. py:function:: safe_float_convert(value)

      Safely convert a value to float or return None if invalid.

      Handles multiple input types including bool, int, float, and str.
      Guarantees no exceptions are raised during conversion.

      :param value: Any value that might be convertible to float.

      :returns: A float value or None if conversion is not possible.
      :rtype: Optional[float]

      .. rubric:: Examples

      >>> safe_float_convert(3.14)
      3.14
      >>> safe_float_convert("3.14")
      3.14
      >>> safe_float_convert(42)
      42.0
      >>> safe_float_convert(True)
      1.0
      >>> safe_float_convert("invalid") is None
      True
      >>> safe_float_convert(None) is None
      True
      >>> safe_float_convert("inf")
      inf
      >>> safe_float_convert("NaN")  # doctest: +ELLIPSIS
      nan

      .. note::

         This function silently handles all conversion errors by returning None.
         Special values like "inf", "-inf", and "nan" are properly handled.


.. py:function:: safe_int_convert(value)

      Safely convert a value to int or return None if invalid.

      Handles multiple input types including bool, int, float, str, and bytes.
      Guarantees no exceptions are raised during conversion.

      :param value: Any value that might be convertible to int.

      :returns: An int value or None if conversion is not possible.
      :rtype: Optional[int]

      .. rubric:: Examples

      >>> safe_int_convert(42)
      42
      >>> safe_int_convert("42")
      42
      >>> safe_int_convert(3.14)
      3
      >>> safe_int_convert(True)
      1
      >>> safe_int_convert("hello") is None
      True
      >>> safe_int_convert(None) is None
      True
      >>> safe_int_convert("0x10")  # Hex strings need explicit handling
      None

      .. note::

         This function silently handles all conversion errors by returning None.
         For hex/octal/binary strings, use the int(x, base) function directly.


.. py:function:: safe_str_convert(value)

      Safely convert a value to string with proper handling of various types.

      Provides special handling for bytes (UTF-8 decoding) and Path objects.
      Never raises exceptions, always returns a valid string.

      :param value: Any value to convert to string.

      :returns: String representation of the value. Empty string for None.
      :rtype: str

      .. rubric:: Examples

      >>> safe_str_convert("hello")
      'hello'
      >>> safe_str_convert(42)
      '42'
      >>> safe_str_convert(None)
      ''
      >>> safe_str_convert(b'hello')
      'hello'
      >>> from pathlib import Path
      >>> safe_str_convert(Path('/tmp'))  # doctest: +SKIP
      '/tmp'
      >>> safe_str_convert(b'\xff\xfe')  # Invalid UTF-8 falls back to str(bytes)
      "b'\xff\xfe'"

      .. note::

         This function attempts UTF-8 decoding for bytes objects and falls back
         to str(bytes) representation if decoding fails.


.. py:function:: try_convert(value, target_type)

      Convert a value to a target type with detailed error reporting.

      Unlike the safe_*_convert functions, this provides structured error information
      when conversion fails rather than just returning None.

      :param value: The value to convert
      :param target_type: The type to convert to

      :returns:

                A result object containing success status, converted
                    value, and error information if conversion failed
      :rtype: ConversionResult[T]

      .. rubric:: Examples

      >>> result = try_convert("42", int)
      >>> result.success
      True
      >>> result.value
      42
      >>> result = try_convert("not_a_number", int)
      >>> result.success
      False
      >>> bool(result)
      False
      >>> result.error is not None
      True
      >>> result = try_convert(None, int)
      >>> result.success
      False

      .. note:: Captures and reports the actual exception that occurred during conversion.


.. py:data:: ConvertibleToBool

.. py:data:: ConvertibleToFloat

.. py:data:: ConvertibleToInt

.. py:data:: ConvertibleToStr

.. py:data:: version
      :value: '0.1.0'


