.. py:module:: src.type_forge.validators.composite


Module Contents
---------------


   .. py:class:: CompositeValidator(validators)   :module: 

      A validator that combines multiple validators.

      Initialize a composite validator with a list of validator functions.

      :param validators: List of validator functions that take a value and return a boolean



.. py:function:: is_in_range(value, min_value, max_value)

      Check if an integer is within a specified range.


.. py:function:: is_not_empty(value)

      Check if a string is not empty.


.. py:function:: is_positive(value)

      Check if an integer is positive.


.. py:function:: is_valid_length(value, min_length = None, max_length = None)

      Check if a collection has a valid length.

      :param value: The collection to check
      :param min_length: Minimum allowed length (inclusive), or None for no minimum
      :param max_length: Maximum allowed length (inclusive), or None for no maximum

      :returns: True if the length is valid, False otherwise


.. py:data:: T

