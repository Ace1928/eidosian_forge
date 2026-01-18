type_forge.typing.analysis
==========================

.. py:module:: type_forge.typing.analysis

.. autoapi-nested-parse::

   Type Analysis for the Type Forge system.

   This module provides utilities for analyzing, comparing, and determining relationships
   between Python types with mathematical precision. It enables systematic type conversion,
   validation, compatibility assessment, and hierarchical analysis.



Classes
-------

.. autoapisummary::

   type_forge.typing.analysis.TypeRelationshipAnalyzer


Module Contents
---------------

.. py:class:: TypeRelationshipAnalyzer

   Analyzes and determines the relationship between types
   for conversion and validation.

   This utility class provides methods to analyze type relationships, determine
   compatibility, and calculate conversion distances between types. It implements
   a hierarchical type analysis system that can determine:

   1. Direct type relationships (identical, subtype, supertype)
   2. Conversion possibilities and their relative complexity
   3. Structural compatibility between collection types
   4. Common supertypes across multiple types

   .. rubric:: Examples

   >>> analyzer = TypeRelationshipAnalyzer()
   >>> analyzer.get_relationship(bool, int)
   <TypeCompatibility.SUBTYPE: 'subtype'>
   >>> analyzer.get_relationship(int, bool)
   <TypeCompatibility.SUPERTYPE: 'supertype'>
   >>> analyzer.get_relationship(int, str)
   <TypeCompatibility.CONVERTIBLE: 'convertible'>
   >>> analyzer.get_relationship(int, int)
   <TypeCompatibility.IDENTICAL: 'identical'>
   >>> analyzer.is_convertible(int, float)
   True
   >>> analyzer.is_convertible(complex, bool)
   False


   .. py:method:: find_common_supertype(*types)

      Find the most specific common supertype of all given types.

      This method analyzes the Method Resolution Order (MRO) of each type to
      identify the most specific type that all given types inherit from. It's
      useful for determining a common interface or base class.

      :param \*types: Variable number of types to analyze

      :returns:

                The most specific common supertype, or None
                                      if only object is common (effectively no
                                      meaningful common interface exists)
      :rtype: Optional[Type[object]]

      .. rubric:: Examples

      >>> analyzer = TypeRelationshipAnalyzer()
      >>> analyzer.find_common_supertype(int, float) == numbers.Number
      True
      >>> analyzer.find_common_supertype(list, tuple) == collections.abc.Sequence
      True
      >>> analyzer.find_common_supertype(str, dict) is None
      True



   .. py:method:: get_conversion_distance(source_type, target_type)

      Calculate the conversion distance between types (lower is easier).

      This method quantifies the complexity of converting between types using
      a numeric distance metric. The distance represents the relative difficulty
      of conversion, with smaller values indicating easier conversions.

      :param source_type: The source type to convert from
      :param target_type: The target type to convert to

      :returns:

                Distance metric with precise meaning:
                    - 0: Identical types (no conversion needed)
                    - 1: Subtype relationship (safe upcast)
                    - 2: Supertype relationship (potential downcast)
                    - 3: Implicit convertible types (automatic conversion)
                    - 5: Explicitly convertible types (requires explicit conversion)
                    - 7: Container compatible types (similar collections)
                    - 10: Structurally compatible types (similar structures)
                    - 15: Protocol compatible types (interface compatibility)
                    - float('inf'): Incompatible types (conversion impossible)
      :rtype: TypeDistance

      .. rubric:: Examples

      >>> analyzer = TypeRelationshipAnalyzer()
      >>> analyzer.get_conversion_distance(int, int)
      0
      >>> analyzer.get_conversion_distance(bool, int)
      1
      >>> analyzer.get_conversion_distance(str, int) >             ...     analyzer.get_conversion_distance(float, int)
      True
      >>> analyzer.get_conversion_distance(list, dict) == float('inf')
      True



   .. py:method:: get_relationship(source_type, target_type)

      Determine the relationship between source and target types.

      This method establishes the fundamental relationship between two types,
      forming the basis for type conversion, validation, and compatibility checks.
      It analyzes inheritance relationships, conversion possibilities, and
      structural compatibilities.

      :param source_type: The source type to analyze
      :param target_type: The target type to analyze

      :returns:

                The precise relationship between the types, with values:
                    - IDENTICAL: Types are exactly the same
                    - SUBTYPE: Source is a subtype of target
                    - SUPERTYPE: Target is a subtype of source
                    - IMPLICIT_CONVERTIBLE: Types can be converted implicitly
                    - CONVERTIBLE: Types can be converted explicitly
                    - CONTAINER_COMPATIBLE: Container types with compatible elements
                    - STRUCTURALLY_COMPATIBLE: Types share compatible structures
                    - PROTOCOL_COMPATIBLE: Source satisfies target's protocol
                    - INCOMPATIBLE: Types cannot be converted
      :rtype: TypeCompatibility

      .. rubric:: Examples

      >>> analyzer = TypeRelationshipAnalyzer()
      >>> analyzer.get_relationship(int, int)
      <TypeCompatibility.IDENTICAL: 'identical'>
      >>> analyzer.get_relationship(bool, int)
      <TypeCompatibility.SUBTYPE: 'subtype'>
      >>> analyzer.get_relationship(list, tuple)
      <TypeCompatibility.CONVERTIBLE: 'convertible'>



   .. py:method:: is_convertible(source_type, target_type)

      Determine if source type can be converted to target type.

      A convenience method that leverages the type relationship analysis to
      provide a boolean verdict on conversion possibility.

      :param source_type: The source type to convert from
      :param target_type: The target type to convert to

      :returns: True if conversion is possible through any means, False otherwise
      :rtype: bool

      .. rubric:: Examples

      >>> analyzer = TypeRelationshipAnalyzer()
      >>> analyzer.is_convertible(int, float)
      True
      >>> analyzer.is_convertible(str, bytes)
      True
      >>> analyzer.is_convertible(dict, list)
      False



