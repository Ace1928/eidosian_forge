type_forge.validators.basic
===========================

.. py:module:: type_forge.validators.basic


Attributes
----------

.. autoapisummary::

   type_forge.validators.basic.T


Classes
-------

.. autoapisummary::

   type_forge.validators.basic.BasicValidator


Module Contents
---------------

.. py:class:: BasicValidator

   Bases: :py:obj:`type_forge.core.base.BaseValidator`


   A class for basic data type validators.


   .. py:method:: safe_bool_convert(value)
      :staticmethod:


      Safely convert a value to bool with semantic interpretation.

      :param value: Any value that might be convertible to bool

      :returns: A bool representation of the value, with common string patterns
                like "yes"/"no" properly handled



   .. py:method:: safe_float_convert(value)
      :staticmethod:


      Safely convert a value to float or return None if invalid.

      :param value: Any value that might be convertible to float

      :returns: A float value or None if conversion is not possible



   .. py:method:: safe_int_convert(value)
      :staticmethod:


      Safely convert a value to int or return None if invalid.

      :param value: Any value that might be convertible to int

      :returns: An int value or None if conversion is not possible



   .. py:method:: safe_str_convert(value)
      :staticmethod:


      Safely convert a value to string with proper handling of various types.

      :param value: Any value to convert to string

      :returns: String representation of the value



   .. py:method:: validate(value)

      Base validation method for the BasicValidator.

      :param value: The value to validate

      :returns: True if validation succeeds, False otherwise



   .. py:method:: validate_boolean(value)
      :staticmethod:


      Validate that the value is a boolean.

      :param value: The value to validate

      :returns: The validated boolean

      :raises ValueError: If the value is not a boolean



   .. py:method:: validate_dict(value, key_type, value_type)
      :staticmethod:


      Validate that the value is a dictionary with specific key and value types.

      :param value: The value to validate
      :param key_type: The expected type of keys in the dictionary
      :param value_type: The expected type of values in the dictionary

      :returns: The validated dictionary

      :raises ValueError: If the value is not a dictionary or contains keys/values of the wrong type



   .. py:method:: validate_float(value)
      :staticmethod:


      Validate that the value is a float.

      :param value: The value to validate

      :returns: The validated float

      :raises ValueError: If the value is not a float



   .. py:method:: validate_integer(value)
      :staticmethod:


      Validate that the value is an integer.

      :param value: The value to validate

      :returns: The validated integer

      :raises ValueError: If the value is not an integer



   .. py:method:: validate_list(value, item_type)
      :staticmethod:


      Validate that the value is a list of a specific item type.

      :param value: The value to validate
      :param item_type: The expected type of items in the list

      :returns: The validated list

      :raises ValueError: If the value is not a list or contains items of the wrong type



   .. py:method:: validate_string(value)
      :staticmethod:


      Validate that the value is a string.

      :param value: The value to validate

      :returns: The validated string

      :raises ValueError: If the value is not a string



   .. py:method:: validate_with_detail(value)

      Validate with detailed information.

      :param value: The value to validate

      :returns: ValidationResult with detailed information



.. py:data:: T

