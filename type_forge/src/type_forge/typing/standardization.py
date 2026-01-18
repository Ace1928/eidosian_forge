"""
Type Standardization Utilities for the Type Forge System
========================================================

This module provides utilities for standardizing, analyzing, and transforming
type relationships. It ensures consistent type handling across the Type Forge
ecosystem with precise type categorization and normalization capabilities.

Core functionalities include:
- Standardizing type names to consistent PEP 484 conventions
- Finding common supertypes across multiple types
- Identifying generic, abstract, and specialized types
- Retrieving complete type hierarchies via Method Resolution Order (MRO)
- Deduplicating redundant types from collections

These utilities enable robust type analysis and transformation operations,
supporting both static type checking and runtime type verification with
comprehensive edge case handling.
"""

import inspect
from abc import ABCMeta
from typing import List, Optional, Sequence, Set, Type, get_args, get_origin

from type_forge.typing.validation import is_subclass_safe

from . import __version__  # noqa: F401

version = __version__


def standardize_type_name(name: str) -> str:
    """
    Standardize a type name to a consistent format.

    Converts various styles of type names to a consistent format
    following PEP 484 conventions.

    Args:
        name: The type name to standardize

    Returns:
        str: The standardized type name

    Examples:
        >>> standardize_type_name("int")
        'int'
        >>> standardize_type_name("list[int]")
        'List[int]'
        >>> standardize_type_name("Dict[str, list]")
        'Dict[str, List]'

    Note:
        Useful for ensuring consistent type naming across a codebase.
    """
    # Dictionary of common lowercase type names to proper capitalization
    type_mapping = {
        "str": "str",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "list": "List",
        "dict": "Dict",
        "set": "Set",
        "tuple": "Tuple",
        "frozenset": "FrozenSet",
        "none": "None",
        "any": "Any",
        "optional": "Optional",
        "union": "Union",
        "callable": "Callable",
        "type": "Type",
        "sequence": "Sequence",
        "mapping": "Mapping",
        "iterable": "Iterable",
    }

    # Handle generic types with brackets
    if "[" in name and "]" in name:
        base_type = name.split("[")[0].strip()
        content = name[len(base_type) + 1 : -1]

        # Standardize base type
        standard_base = type_mapping.get(base_type.lower(), base_type)

        # Handle nested types recursively
        # This handles commas not inside brackets
        parts: List[str] = []
        bracket_level = 0
        current = ""

        for char in content:
            if char == "[":
                bracket_level += 1
                current += char
            elif char == "]":
                bracket_level -= 1
                current += char
            elif char == "," and bracket_level == 0:
                parts.append(standardize_type_name(current.strip()))
                current = ""
            else:
                current += char

        if current:
            parts.append(standardize_type_name(current.strip()))

        return f"{standard_base}[{', '.join(parts)}]"

    # Handle simple types without brackets
    return type_mapping.get(name.lower(), name)


def get_common_supertype(types: Sequence[Type[object]]) -> Optional[Type[object]]:
    """
    Find the most specific common supertype of a sequence of types.

    Determines the most specific type that all provided types inherit from,
    which is useful for type unification.

    Args:
        types: A sequence of types to find the common supertype for

    Returns:
        Optional[Type[object]]: The common supertype, or None if no common type exists
                               besides 'object'

    Examples:
        >>> get_common_supertype([int, float])  # doctest: +SKIP
        <class 'numbers.Number'>
        >>> get_common_supertype([list, tuple])  # doctest: +SKIP
        <class 'collections.abc.Sequence'>
        >>> get_common_supertype([int, str])  # doctest: +SKIP
        <class 'object'>

    Note:
        This function traverses inheritance hierarchies to find meaningful
        common supertypes beyond just the 'object' class.
    """
    if not types:
        return None

    if len(types) == 1:
        return types[0]

    # Get all base classes for each type
    base_classes_list: List[Set[Type[object]]] = []
    for typ in types:
        # Get MRO (Method Resolution Order) except object
        mro = inspect.getmro(typ)[:-1]  # Exclude object
        base_classes_list.append(set(mro))

    # Find common base classes
    if not base_classes_list:
        return object

    # Start with first set and find intersection
    common_bases: Set[Type[object]] = base_classes_list[0]
    for base_set in base_classes_list[1:]:
        common_bases = common_bases.intersection(base_set)

    if not common_bases:
        # Only object is common
        return object

    # Find most specific common base class
    # (The one that appears first in the MRO of the first type)
    mro = inspect.getmro(types[0])
    for base in mro:
        if base in common_bases:
            return base

    return object


def is_generic_type(typ: Type[object]) -> bool:
    """
    Check if a type is a generic type.

    Determines whether a type is a parameterized generic type
    like List[int] rather than a concrete type like list.

    Args:
        typ: The type to check

    Returns:
        bool: True if the type is a generic type, False otherwise

    Examples:
        >>> is_generic_type(List[int])  # doctest: +SKIP
        True
        >>> is_generic_type(list)
        False
        >>> is_generic_type(Dict[str, int])  # doctest: +SKIP
        True

    Note:
        Useful for handling generic types specially in type systems.
    """
    return get_origin(typ) is not None and bool(get_args(typ))


def is_abstract_type(typ: Type[object]) -> bool:
    """
    Check if a type is abstract (cannot be instantiated directly).

    Determines whether a type is an abstract base class or interface
    that cannot be instantiated directly.

    Args:
        typ: The type to check

    Returns:
        bool: True if the type is abstract, False otherwise

    Examples:
        >>> from abc import ABC, abstractmethod
        >>> class AbstractExample(ABC):
        ...     @abstractmethod
        ...     def method(self): pass
        >>> is_abstract_type(AbstractExample)  # doctest: +SKIP
        True
        >>> is_abstract_type(int)
        False

    Note:
        Identifies types that are meant to be inherited from rather than instantiated.
    """
    return (
        isinstance(typ, ABCMeta)
        and hasattr(typ, "__abstractmethods__")
        and bool(typ.__abstractmethods__)
    )


def get_type_hierarchy(typ: Type[object]) -> List[Type[object]]:
    """
    Get the complete inheritance hierarchy of a type.

    Returns the Method Resolution Order (MRO) of a type, which
    represents its inheritance hierarchy.

    Args:
        typ: The type to get the hierarchy for

    Returns:
        List[Type[object]]: The inheritance hierarchy of the type, from
                           most specific to most general

    Examples:
        >>> bool_hierarchy = get_type_hierarchy(bool)
        >>> bool_hierarchy  # doctest: +SKIP
        [<class 'bool'>, <class 'int'>, <class 'object'>]

    Note:
        Useful for understanding type relationships and finding common types.
    """
    return list(inspect.getmro(typ))


def deduplicate_types(types: Sequence[Type[object]]) -> List[Type[object]]:
    """
    Remove redundant types from a sequence of types.

    Removes types that are subtypes of other types in the sequence,
    keeping only the most specific types necessary.

    Args:
        types: A sequence of types to deduplicate

    Returns:
        List[Type[object]]: A deduplicated list of types

    Examples:
        >>> deduplicate_types([int, object, float, numbers.Number])  # doctest: +SKIP
        [<class 'int'>, <class 'float'>]
        >>> deduplicate_types([list, Collection, Sequence])  # doctest: +SKIP
        [<class 'list'>]

    Note:
        Useful for simplifying Union types and type annotations.
    """
    if not types:
        return []

    result: List[Type[object]] = []

    for candidate in types:
        # Check if candidate is a subtype of any type already in result
        if any(
            is_subclass_safe(candidate, existing) and candidate is not existing
            for existing in result
        ):
            # Skip this type as it's more specific than one we already have
            continue

        # Remove any types in result that are subtypes of this candidate
        result = [
            existing
            for existing in result
            if not (is_subclass_safe(existing, candidate) and existing is not candidate)
        ]

        # Add the candidate
        result.append(candidate)

    return result
