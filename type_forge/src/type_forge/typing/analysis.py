"""
Type Analysis for the Type Forge system.

This module provides utilities for analyzing, comparing, and determining relationships
between Python types with mathematical precision. It enables systematic type conversion,
validation, compatibility assessment, and hierarchical analysis.
"""

import inspect
from typing import List, Optional, Type, cast

from .aliases import CollectionTypes, NumericTypes, PrimitiveTypes, TypeDistance
from .definitions import TypeCompatibility
from .variables import T, U


class TypeRelationshipAnalyzer:
    """
    Analyzes and determines the relationship between types
    for conversion and validation.

    This utility class provides methods to analyze type relationships, determine
    compatibility, and calculate conversion distances between types. It implements
    a hierarchical type analysis system that can determine:

    1. Direct type relationships (identical, subtype, supertype)
    2. Conversion possibilities and their relative complexity
    3. Structural compatibility between collection types
    4. Common supertypes across multiple types

    Examples:
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
    """

    def get_relationship(
        self,
        source_type: Type[T],
        target_type: Type[U],
    ) -> TypeCompatibility:
        """
        Determine the relationship between source and target types.

        This method establishes the fundamental relationship between two types,
        forming the basis for type conversion, validation, and compatibility checks.
        It analyzes inheritance relationships, conversion possibilities, and
        structural compatibilities.

        Args:
            source_type: The source type to analyze
            target_type: The target type to analyze

        Returns:
            TypeCompatibility: The precise relationship between the types, with values:
                - IDENTICAL: Types are exactly the same
                - SUBTYPE: Source is a subtype of target
                - SUPERTYPE: Target is a subtype of source
                - IMPLICIT_CONVERTIBLE: Types can be converted implicitly
                - CONVERTIBLE: Types can be converted explicitly
                - CONTAINER_COMPATIBLE: Container types with compatible elements
                - STRUCTURALLY_COMPATIBLE: Types share compatible structures
                - PROTOCOL_COMPATIBLE: Source satisfies target's protocol
                - INCOMPATIBLE: Types cannot be converted

        Examples:
            >>> analyzer = TypeRelationshipAnalyzer()
            >>> analyzer.get_relationship(int, int)
            <TypeCompatibility.IDENTICAL: 'identical'>
            >>> analyzer.get_relationship(bool, int)
            <TypeCompatibility.SUBTYPE: 'subtype'>
            >>> analyzer.get_relationship(list, tuple)
            <TypeCompatibility.CONVERTIBLE: 'convertible'>
        """
        # Identical types
        if source_type is target_type:
            return TypeCompatibility.IDENTICAL

        # Subtype relationship
        if issubclass(source_type, target_type):
            return TypeCompatibility.SUBTYPE

        # Supertype relationship
        if issubclass(target_type, source_type):
            return TypeCompatibility.SUPERTYPE

        # Common conversions between primitive types
        if source_type in PrimitiveTypes and target_type in PrimitiveTypes:
            # Most numeric types can be converted
            if source_type in NumericTypes and target_type in NumericTypes:
                return TypeCompatibility.IMPLICIT_CONVERTIBLE

            # String representations
            if target_type is str:
                return TypeCompatibility.CONVERTIBLE

            # String to numeric conversions
            if source_type is str and target_type in NumericTypes:
                return TypeCompatibility.CONVERTIBLE

        # Container type conversions
        if source_type in CollectionTypes and target_type in CollectionTypes:
            # Similar collection types are often convertible
            return TypeCompatibility.CONTAINER_COMPATIBLE

        # Default to incompatible
        return TypeCompatibility.INCOMPATIBLE

    def get_conversion_distance(
        self,
        source_type: Type[T],
        target_type: Type[U],
    ) -> TypeDistance:
        """
        Calculate the conversion distance between types (lower is easier).

        This method quantifies the complexity of converting between types using
        a numeric distance metric. The distance represents the relative difficulty
        of conversion, with smaller values indicating easier conversions.

        Args:
            source_type: The source type to convert from
            target_type: The target type to convert to

        Returns:
            TypeDistance: Distance metric with precise meaning:
                - 0: Identical types (no conversion needed)
                - 1: Subtype relationship (safe upcast)
                - 2: Supertype relationship (potential downcast)
                - 3: Implicit convertible types (automatic conversion)
                - 5: Explicitly convertible types (requires explicit conversion)
                - 7: Container compatible types (similar collections)
                - 10: Structurally compatible types (similar structures)
                - 15: Protocol compatible types (interface compatibility)
                - float('inf'): Incompatible types (conversion impossible)

        Examples:
            >>> analyzer = TypeRelationshipAnalyzer()
            >>> analyzer.get_conversion_distance(int, int)
            0
            >>> analyzer.get_conversion_distance(bool, int)
            1
            >>> analyzer.get_conversion_distance(str, int) > \
            ...     analyzer.get_conversion_distance(float, int)
            True
            >>> analyzer.get_conversion_distance(list, dict) == float('inf')
            True
        """
        relationship = self.get_relationship(source_type, target_type)

        if relationship == TypeCompatibility.IDENTICAL:
            return 0
        if relationship == TypeCompatibility.SUBTYPE:
            return 1
        if relationship == TypeCompatibility.SUPERTYPE:
            return 2
        if relationship == TypeCompatibility.IMPLICIT_CONVERTIBLE:
            return 3
        if relationship == TypeCompatibility.CONVERTIBLE:
            return 5
        if relationship == TypeCompatibility.CONTAINER_COMPATIBLE:
            return 7
        if relationship == TypeCompatibility.STRUCTURALLY_COMPATIBLE:
            return 10
        if relationship == TypeCompatibility.PROTOCOL_COMPATIBLE:
            return 15
        return cast(TypeDistance, float("inf"))

    def is_convertible(self, source_type: Type[T], target_type: Type[U]) -> bool:
        """
        Determine if source type can be converted to target type.

        A convenience method that leverages the type relationship analysis to
        provide a boolean verdict on conversion possibility.

        Args:
            source_type: The source type to convert from
            target_type: The target type to convert to

        Returns:
            bool: True if conversion is possible through any means, False otherwise

        Examples:
            >>> analyzer = TypeRelationshipAnalyzer()
            >>> analyzer.is_convertible(int, float)
            True
            >>> analyzer.is_convertible(str, bytes)
            True
            >>> analyzer.is_convertible(dict, list)
            False
        """
        relationship = self.get_relationship(source_type, target_type)
        return relationship.is_compatible()

    def find_common_supertype(self, *types: Type[object]) -> Optional[Type[object]]:
        """
        Find the most specific common supertype of all given types.

        This method analyzes the Method Resolution Order (MRO) of each type to
        identify the most specific type that all given types inherit from. It's
        useful for determining a common interface or base class.

        Args:
            *types: Variable number of types to analyze

        Returns:
            Optional[Type[object]]: The most specific common supertype, or None
                                  if only object is common (effectively no
                                  meaningful common interface exists)

        Examples:
            >>> analyzer = TypeRelationshipAnalyzer()
            >>> analyzer.find_common_supertype(int, float) == numbers.Number
            True
            >>> analyzer.find_common_supertype(list, tuple) == collections.abc.Sequence
            True
            >>> analyzer.find_common_supertype(str, dict) is None
            True
        """
        if not types:
            return None

        # Start with the first type's MRO (Method Resolution Order)
        common_mro: List[Type[object]] = list(inspect.getmro(types[0]))

        # Intersect with MROs of all other types
        for t in types[1:]:
            t_mro = inspect.getmro(t)
            common_mro = [cls for cls in common_mro if cls in t_mro]

        # object is always common, so exclude it if it's the only common ancestor
        if len(common_mro) == 1 and common_mro[0] is object:
            return None

        # Return the most specific common type (first in MRO)
        return common_mro[0]
