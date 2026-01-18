"""
Type Naming Utilities for the Type Forge System
===============================================

This module provides utilities for obtaining standardized names and
representations of Python types. It ensures consistent naming conventions
and human-readable type descriptions across the Type Forge system.

Core functionalities include:
- Generating standardized type names for both built-in and custom types
- Handling complex generic types with nested parameters
- Identifying primitive types for special treatment
- Providing clean string representations of type hierarchies
- Converting between different type representation formats
- Analyzing type structures for naming consistency

These utilities form the foundation for consistent type naming and
presentation throughout the Type Forge ecosystem, supporting both
programmatic type operations and human-readable documentation.
"""

import inspect
from types import NoneType
from typing import (
    Any,
    Dict,
    Final,
    Literal,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from type_forge.typing.aliases import CollectionTypes, PrimitiveTypes

from . import __version__  # noqa: F401

# Version information
version: Final[str] = __version__

# Type variables with proper constraints
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

# Type category literal for strict typing
TypeCategoryLiteral = Literal[
    "primitive", "container", "composite", "callable", "special"
]

# Constants
PRIMITIVE_CATEGORY: TypeCategoryLiteral = "primitive"
CONTAINER_CATEGORY: TypeCategoryLiteral = "container"
COMPOSITE_CATEGORY: TypeCategoryLiteral = "composite"
CALLABLE_CATEGORY: TypeCategoryLiteral = "callable"
SPECIAL_CATEGORY: TypeCategoryLiteral = "special"

# Use NoneType correctly for type comparisons
_NONE_TYPE = NoneType


class TypeProtocol(Protocol):
    """Protocol defining the interface for type objects."""

    __name__: str
    __module__: str
    __qualname__: str


def get_type_name(typ: Any) -> str:
    """
    Get the name of a type.

    For simple types, returns the type name.
    For generic types, includes the parameter types.

    Args:
        typ: The type to get the name for

    Returns:
        str: The name of the type

    Examples:
        >>> get_type_name(int)
        'int'
        >>> from typing import List
        >>> get_type_name(List[int])
        'List[int]'
    """
    # Handle None type
    if typ is _NONE_TYPE:
        return "None"

    # Handle generic types
    if is_generic_type(typ):
        origin = get_origin(typ)
        args = get_args(typ)

        if origin is Union:
            # Handle Union types specially
            if len(args) == 2 and args[1] is _NONE_TYPE:
                # This is Optional[T]
                return f"Optional[{get_type_name(args[0])}]"
            return f"Union[{', '.join(get_type_name(arg) for arg in args)}]"

        # Get the name of the origin
        if hasattr(origin, "__name__"):
            origin_name = origin.__name__
        else:
            # Fall back to string representation if __name__ is not available
            origin_name = str(origin).replace("typing.", "")

        if not args:
            return origin_name

        # Format the type arguments
        args_str = ", ".join(get_type_name(arg) for arg in args)
        return f"{origin_name}[{args_str}]"

    # Handle simple types
    if hasattr(typ, "__name__"):
        return typ.__name__

    # Fall back to string representation
    return str(typ)


def is_primitive_type(typ: Union[type, Any]) -> bool:
    """
    Check if a type is primitive.

    Primitive types are basic types like int, str, float, bool, etc.

    Args:
        typ: The type to check

    Returns:
        bool: True if the type is primitive, False otherwise
    """
    return typ in PrimitiveTypes


def is_container_type(typ: Union[type, Any]) -> bool:
    """
    Check if a type is a container.

    Container types include list, dict, set, tuple, etc.

    Args:
        typ: The type to check

    Returns:
        bool: True if the type is a container, False otherwise
    """
    # Built-in container types
    if typ in CollectionTypes:
        return True

    # Check for generic container
    origin = get_origin(typ)
    if origin is not None:
        return origin in CollectionTypes

    # Check for container protocol implementation
    try:
        return hasattr(typ, "__iter__") and hasattr(typ, "__len__")
    except (TypeError, AttributeError):
        return False


def is_generic_type(typ: Any) -> bool:
    """
    Check if a type is generic.

    Generic types include List[T], Dict[K, V], etc.

    Args:
        typ: The type to check

    Returns:
        bool: True if the type is generic, False otherwise

    Examples:
        >>> from typing import List
        >>> is_generic_type(List[int])
        True
        >>> is_generic_type(int)
        False
    """
    return get_origin(typ) is not None


def get_type_category(typ: Any) -> TypeCategoryLiteral:
    """
    Get the category of a type.

    Categorizes a type into one of: primitive, container, composite, callable, or special.

    Args:
        typ: The type to categorize

    Returns:
        TypeCategoryLiteral: The category of the type

    Examples:
        >>> get_type_category(int)
        'primitive'
        >>> get_type_category(list)
        'container'
        >>> get_type_category(Union[int, str])
        'composite'
        >>> get_type_category(Callable[[int], str])
        'callable'

    Note:
        Categories help determine how types should be processed or displayed.
    """
    if is_primitive_type(typ):
        return PRIMITIVE_CATEGORY

    if is_container_type(typ):
        return CONTAINER_CATEGORY

    if callable(typ) or (
        is_generic_type(typ) and str(get_origin(typ)).endswith("Callable")
    ):
        return CALLABLE_CATEGORY

    if is_generic_type(typ) and get_origin(typ) is Union:
        return COMPOSITE_CATEGORY

    return SPECIAL_CATEGORY


def get_generic_args(typ: Any) -> Tuple[Any, ...]:
    """
    Get the type arguments of a generic type.

    For generic types like List[int], returns the inner types (int).

    Args:
        typ: The generic type to inspect

    Returns:
        Tuple[Any, ...]: The type arguments of the generic type

    Examples:
        >>> from typing import List, Dict
        >>> get_generic_args(List[int])
        (int,)
        >>> get_generic_args(Dict[str, int])
        (str, int)
        >>> get_generic_args(int)
        ()

    Note:
        Returns an empty tuple for non-generic types.
    """
    if not is_generic_type(typ):
        return tuple()

    return get_args(typ)


def is_optional_type(typ: Any) -> bool:
    """
    Check if a type is Optional[T].

    Detects if a type is Union[T, None] or Optional[T].

    Args:
        typ: The type to check

    Returns:
        bool: True if the type is Optional, False otherwise

    Examples:
        >>> from typing import Optional
        >>> is_optional_type(Optional[int])
        True
        >>> is_optional_type(int)
        False
        >>> is_optional_type(Union[str, None])
        True

    Note:
        Optional[T] is equivalent to Union[T, None].
    """
    origin = get_origin(typ)
    if origin is not Union:
        return False

    args = get_args(typ)
    return len(args) == 2 and args[1] is _NONE_TYPE


def get_fully_qualified_name(typ: Any) -> str:
    """
    Get the fully qualified name of a type.

    Returns the complete module path and type name.

    Args:
        typ: The type to get the name for

    Returns:
        str: The fully qualified name of the type

    Examples:
        >>> get_fully_qualified_name(int)
        'builtins.int'
        >>> import collections
        >>> get_fully_qualified_name(collections.defaultdict)  # doctest: +SKIP
        'collections.defaultdict'

    Note:
        Useful for serialization and type lookup across module boundaries.
    """
    if is_generic_type(typ):
        origin = get_origin(typ)
        args = get_args(typ)

        if origin is None:
            return str(typ)

        origin_name = get_fully_qualified_name(origin)

        if not args:
            return origin_name

        args_str = ", ".join(get_fully_qualified_name(arg) for arg in args)
        return f"{origin_name}[{args_str}]"

    if hasattr(typ, "__module__") and hasattr(typ, "__qualname__"):
        if typ.__module__ == "builtins":
            return typ.__qualname__
        return f"{typ.__module__}.{typ.__qualname__}"

    return str(typ)


def format_type_annotation(typ: Any, for_docstring: bool = False) -> str:
    """
    Format a type for use in annotations or docstrings.

    Generates properly formatted type strings for different contexts.

    Args:
        typ: The type to format
        for_docstring: Whether the type is being formatted for a docstring

    Returns:
        str: The formatted type annotation

    Examples:
        >>> from typing import List, Optional
        >>> format_type_annotation(List[int])
        'List[int]'
        >>> format_type_annotation(Optional[str], for_docstring=True)
        'str or None'

    Note:
        When formatting for docstrings, uses more human-readable forms.
    """
    if for_docstring:
        # More readable format for docstrings
        if is_optional_type(typ):
            args = get_args(typ)
            return f"{get_type_name(args[0])} or None"

        if is_generic_type(typ) and get_origin(typ) is Union:
            args = get_args(typ)
            return " or ".join(get_type_name(arg) for arg in args)

    return get_type_name(typ)


def are_types_compatible(source_type: Any, target_type: Any) -> bool:
    """
    Check if two types are compatible.

    Determines if a value of source_type can be used where target_type is expected.

    Args:
        source_type: The source type to check
        target_type: The target type to check against

    Returns:
        bool: True if the types are compatible, False otherwise

    Examples:
        >>> are_types_compatible(int, float)
        True
        >>> from typing import List
        >>> are_types_compatible(List[int], List[float])
        False
        >>> from typing import Any
        >>> are_types_compatible(str, Any)
        True

    Note:
        Compatibility checks consider subclass relationships and type hierarchies.
    """
    # Any type is compatible with itself
    if source_type == target_type:
        return True

    # Handle special cases for Any type
    if str(target_type).endswith("Any"):
        return True

    # Handle optional types
    if is_optional_type(target_type):
        args = get_args(target_type)
        if source_type is _NONE_TYPE:
            return True
        return are_types_compatible(source_type, args[0])

    # Handle basic subclass relationships
    try:
        if not is_generic_type(source_type) and not is_generic_type(target_type):
            if inspect.isclass(source_type) and inspect.isclass(target_type):
                return issubclass(source_type, target_type)
    except TypeError:
        pass

    # Handle generic types
    if is_generic_type(source_type) and is_generic_type(target_type):
        source_origin = get_origin(source_type)
        target_origin = get_origin(target_type)

        if source_origin is not target_origin:
            return False

        source_args = get_args(source_type)
        target_args = get_args(target_type)

        if len(source_args) != len(target_args):
            return False

        # Simple check for covariance - can be extended for proper variance rules
        return all(
            are_types_compatible(source_arg, target_arg)
            for source_arg, target_arg in zip(source_args, target_args)
        )

    # Fallback
    return False


def normalize_type(typ: Any) -> Any:
    """
    Normalize a type representation.

    Converts type aliases to their canonical form for consistent representation.

    Args:
        typ: The type to normalize

    Returns:
        Any: The normalized type

    Examples:
        >>> from typing import List, Sequence
        >>> normalize_type(List[int]) == list[int]  # Python 3.9+ syntax
        True

    Note:
        Normalization ensures consistent type representations across the system.
    """
    # Handle None type
    if typ is _NONE_TYPE:
        return _NONE_TYPE

    # Handle generics
    if is_generic_type(typ):
        origin = get_origin(typ)
        args = get_args(typ)

        if origin is not None and args:
            normalized_args = tuple(normalize_type(arg) for arg in args)
            try:
                return origin[normalized_args]
            except (TypeError, IndexError):
                # Fall back if we can't reconstruct the generic
                return typ

    return typ


def describe_type(typ: Any) -> Dict[str, object]:
    """
    Get a detailed description of a type.

    Returns a dictionary with comprehensive type information.

    Args:
        typ: The type to describe

    Returns:
        Dict[str, object]: A dictionary containing type details

    Examples:
        >>> details = describe_type(int)
        >>> details['name']
        'int'
        >>> details['category']
        'primitive'

    Note:
        Provides deep type structure analysis for complex inspections.
    """
    result: Dict[str, object] = {
        "name": get_type_name(typ),
        "qualified_name": get_fully_qualified_name(typ),
        "category": get_type_category(typ),
        "is_primitive": is_primitive_type(typ),
        "is_container": is_container_type(typ),
        "is_generic": is_generic_type(typ),
        "is_optional": is_optional_type(typ),
    }

    # Add generic arguments if applicable
    if is_generic_type(typ):
        args = get_generic_args(typ)
        result["origin"] = get_origin(typ)
        result["args"] = [describe_type(arg) for arg in args]

    # Add module information if available
    if hasattr(typ, "__module__"):
        result["module"] = typ.__module__

    return result
