type_forge.validators.composite
===============================

.. py:module:: type_forge.validators.composite


Attributes
----------

.. autoapisummary::

   type_forge.validators.composite.T


Classes
-------

.. autoapisummary::

   type_forge.validators.composite.CompositeValidator


Functions
---------

.. autoapisummary::

   type_forge.validators.composite.is_in_range
   type_forge.validators.composite.is_not_empty
   type_forge.validators.composite.is_positive
   type_forge.validators.composite.is_valid_length


Module Contents
---------------

.. py:class:: CompositeValidator(validators)

   Bases: :py:obj:`type_forge.core.base.BaseValidator`, :py:obj:`Generic`\ [\ :py:obj:`T`\ ]


   A validator that combines multiple validators.

   Initialize a composite validator with a list of validator functions.

   :param validators: List of validator functions that take a value and return a boolean


   .. py:method:: add_validator(validator)

      Add a new validator to the composite.

      :param validator: A function that takes a value and returns a boolean



   .. py:method:: from_validators(validators)
      :staticmethod:


      Create a CompositeValidator from a sequence of BaseValidator instances.

      :param validators: A sequence of BaseValidator instances

      :returns: A new CompositeValidator that aggregates all the validators



   .. py:method:: validate(value)

      Validate a value using all validators in the composite.

      :param value: The value to validate

      :returns: True if all validators return True, False otherwise



   .. py:method:: validate_with_details(value, path = '$')

      Validate with detailed result information.

      :param value: The value to validate
      :param path: The path to report in case of violations

      :returns: A ValidationResult with detailed information



   .. py:attribute:: validators


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

