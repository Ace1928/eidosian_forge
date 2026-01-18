.. py:module:: src.type_forge.typing.analysis

Type Analysis for the Type Forge system.

   This module provides utilities for analyzing, comparing, and determining relationships
   between Python types with mathematical precision. It enables systematic type conversion,
   validation, compatibility assessment, and hierarchical analysis.


Module Contents
---------------


   .. py:class:: TypeRelationshipAnalyzer   :module: 

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



