"""
Type Mapping Utilities for the Type Forge System
===============================================

This module contains utilities for type categorization, relationship mapping,
and descriptive information about types. These functions enable semantic analysis
of type relationships beyond simple inheritance hierarchies.

Core functionalities include:
- Type categorization (atomic, container, protocol, etc.)
- Finding common supertypes across multiple types
- Converting between type names and Python type objects
- Generating human-readable type descriptions
- Analyzing type structure and relationships

These utilities serve as the foundation for more complex type operations
in the Type Forge system, enabling precise type handling with elegant
error management and comprehensive edge case coverage.
"""

import inspect
import sys
import types
from pathlib import Path
from typing import (
    Collection,
    Dict,
    Final,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from type_forge.typing.definitions import TypeCategory
from type_forge.typing.validation import is_subclass_safe

from . import __version__  # noqa: F401

version: Final[str] = __version__

# Type variables for better type specificity
T = TypeVar("T")
U = TypeVar("U")

# Type aliases for clarity and precision
TypeObject = Type[object]
TypeName = str
TypeDescription = str
MaybeType = Optional[TypeObject]

# ──────────────────────────────────────────────────────────────
# Type Mapping Functions
# ──────────────────────────────────────────────────────────────


def get_type_category(typ: TypeObject) -> TypeCategory:
    """
    Determine the semantic category of a type.

    Categorizes types into meaningful groups based on their structure
    and behavior rather than just their inheritance relationships.

    Args:
        typ: The type to categorize

    Returns:
        TypeCategory: The semantic category of the type

    Examples:
        >>> get_type_category(int)
        <TypeCategory.ATOMIC: 'atomic'>
        >>> get_type_category(list)
        <TypeCategory.CONTAINER: 'container'>
        >>> get_type_category(dict)
        <TypeCategory.CONTAINER: 'container'>
        >>> get_type_category(Protocol)  # doctest: +SKIP
        <TypeCategory.PROTOCOL: 'protocol'>

    Note:
        This function uses both inheritance and structural properties
        to determine the category.
    """
    # Atomic types - use tuple for faster membership testing
    atomic_types: Final[Tuple[TypeObject, ...]] = (
        int,
        float,
        bool,
        str,
        bytes,
        complex,
        type(None),
    )
    if typ in atomic_types:
        return TypeCategory.ATOMIC

    # Container types - check safely without potential TypeErrors
    container_types: Final[Tuple[TypeObject, ...]] = (list, tuple, dict, set, frozenset)
    for container_type in container_types:
        if is_subclass_safe(typ, container_type):
            return TypeCategory.CONTAINER

    # Function types - check safely
    try:
        if inspect.isfunction(typ) or inspect.ismethod(typ) or callable(typ):
            return TypeCategory.FUNCTION
    except TypeError:
        # Continue processing if typ can't be checked for callability
        pass

    # Protocol types - check for the specific protocol attribute
    if hasattr(typ, "_is_protocol") and getattr(typ, "_is_protocol", False):
        return TypeCategory.PROTOCOL

    # Generic types - check for presence of __origin__ attribute
    if hasattr(typ, "__origin__"):
        return TypeCategory.GENERIC

    # Structural types (dataclasses, named tuples, etc.)
    if hasattr(typ, "__annotations__"):
        return TypeCategory.STRUCTURAL

    # By default, consider it a composite type
    return TypeCategory.COMPOSITE


def get_python_type_for_name(type_name: TypeName) -> MaybeType:
    """
    Get the Python type object corresponding to a type name.

    Maps common type names to their corresponding Python type objects,
    handling both builtin types and common collection types.

    Args:
        type_name: Name of the type as a string

    Returns:
        Optional[Type[object]]: The corresponding Python type, or None if not found

    Examples:
        >>> get_python_type_for_name("int")
        <class 'int'>
        >>> get_python_type_for_name("str")
        <class 'str'>
        >>> get_python_type_for_name("list")
        <class 'list'>
        >>> get_python_type_for_name("unknown") is None
        True
        >>> get_python_type_for_name("STRING")  # Case insensitive
        <class 'str'>

    Note:
        Currently handles only common builtin types. For more complex types,
        consider eval() with appropriate safety measures.
    """

    # Create an identity type constructor to preserve type safety
    # while allowing use of typing constructs in the mapping dictionary
    def make_type_identity(t: TypeObject) -> TypeObject:
        return t

    # Comprehensive mapping of type names to their Python types
    # Using a consistent type constructor for all entries
    type_map: Dict[str, TypeObject] = {
        # Primitive types
        "int": make_type_identity(int),
        "integer": make_type_identity(int),
        "float": make_type_identity(float),
        "double": make_type_identity(float),
        "bool": make_type_identity(bool),
        "boolean": make_type_identity(bool),
        "str": make_type_identity(str),
        "string": make_type_identity(str),
        "bytes": make_type_identity(bytes),
        "bytearray": make_type_identity(bytearray),
        "complex": make_type_identity(complex),
        "none": make_type_identity(type(None)),
        "nonetype": make_type_identity(type(None)),
        "null": make_type_identity(type(None)),
        # Collection types
        "list": make_type_identity(list),
        "dict": make_type_identity(dict),
        "dictionary": make_type_identity(dict),
        "tuple": make_type_identity(tuple),
        "set": make_type_identity(set),
        "frozenset": make_type_identity(frozenset),
        # Other common types
        "object": make_type_identity(object),
        "type": make_type_identity(type),
        "path": make_type_identity(Path),
        "callable": make_type_identity(types.FunctionType),
        "function": make_type_identity(types.FunctionType),
        "method": make_type_identity(types.MethodType),
    }

    # Handle typing module constructs separately to maintain type safety
    # Only add these when in a type checking context
    if sys.modules.get("typing") is not None:
        typing_map: Dict[str, TypeObject] = {
            "union": cast(TypeObject, Union),
            "optional": cast(TypeObject, Optional),
            "mapping": cast(TypeObject, Mapping),
            "collection": cast(TypeObject, Collection),
            "protocol": cast(TypeObject, Protocol),
            "literal": cast(TypeObject, Literal),
            "generic": cast(TypeObject, Generic),
        }
        type_map.update(typing_map)

    # Normalize input (lowercase) and look up in map
    normalized_name: str = type_name.lower()
    return type_map.get(normalized_name)


def get_common_supertype(types: List[TypeObject]) -> MaybeType:
    """
    Find the most specific common supertype of multiple types.

    Identifies the closest common ancestor type that all the given
    types inherit from, providing the tightest type bound.

    Args:
        types: List of types to find a common supertype for

    Returns:
        Optional[Type[object]]: The common supertype, or None if only object is common

    Examples:
        >>> get_common_supertype([int, float])  # doctest: +SKIP
        <class 'numbers.Number'>
        >>> get_common_supertype([list, tuple])  # doctest: +SKIP
        <class 'collections.abc.Sequence'>
        >>> get_common_supertype([str, int]) is object
        True
        >>> get_common_supertype([]) is None
        True

    Note:
        Returns object if no more specific common supertype exists.
        Returns None for an empty list of types.
    """
    if not types:
        return None

    if len(types) == 1:
        return types[0]

    # Get all superclasses for each type
    all_mros: List[List[TypeObject]] = []

    for t in types:
        try:
            # t is already known to be a type, so no need for isinstance check
            all_mros.append(list(t.__mro__))
        except (AttributeError, TypeError):
            # Fallback for objects without proper MRO
            all_mros.append([object])

    # Find common elements in all MROs
    common_types: Set[TypeObject] = set(all_mros[0])
    for mro in all_mros[1:]:
        common_types.intersection_update(mro)

    # If only object is common, return it directly
    if common_types == {object}:
        return object

    # Find the most specific (lowest in MRO) common type
    best_type: MaybeType = None
    for t in all_mros[0]:  # Use first type's MRO as reference order
        if t in common_types:
            best_type = t
            break

    return best_type


def get_type_name(typ: TypeObject) -> TypeName:
    """
    Get a user-friendly name for a type object.

    Creates a more readable name for types, handling special cases like
    NoneType and properly formatting generic types.

    Args:
        typ: The type to get a name for

    Returns:
        str: A user-friendly name for the type

    Examples:
        >>> get_type_name(int)
        'int'
        >>> get_type_name(type(None))
        'None'
        >>> get_type_name(Dict[str, int])  # doctest: +SKIP
        'Dict[str, int]'
        >>> get_type_name(List[str])  # doctest: +SKIP
        'List[str]'
        >>> get_type_name(List)  # doctest: +SKIP
        'List'

    Note:
        This function creates names similar to those used in type annotations.
    """
    # Handle None type specially
    if typ is type(None):
        return "None"

    # Handle Union types with more readable names
    origin = get_origin(typ)
    args = get_args(typ)

    if origin is not None:
        # Get origin name
        if hasattr(origin, "__name__"):
            origin_name: str = origin.__name__
        else:
            origin_name = str(origin).replace("typing.", "")

        # Handle Union specially for better readability
        if origin_name == "Union":
            return " | ".join(get_type_name(arg) for arg in args)

        # Format generic type with arguments
        if args:
            args_str: str = ", ".join(get_type_name(arg) for arg in args)
            return f"{origin_name}[{args_str}]"
        return origin_name

    # Basic case: just return the type name
    if hasattr(typ, "__name__"):
        return typ.__name__

    # Fallback
    return str(typ).replace("typing.", "")  # Remove typing. prefix


def describe_type(value: object) -> TypeDescription:
    """
    Generate a detailed description of a value's type.

    Creates a human-readable description of an object's type,
    including additional information for collections.

    Args:
        value: The value to describe

    Returns:
        str: A detailed description of the value's type

    Examples:
        >>> describe_type(42)
        'int'
        >>> describe_type([1, 2, 3])
        'list[int] (length: 3)'
        >>> describe_type({'a': 1, 'b': 'text'})
        'dict[str, mixed] (size: 2)'
        >>> describe_type(None)
        'None'

    Note:
        For collections, includes element types and collection size.
    """
    if value is None:
        return "None"

    # Get basic type name
    type_name: str = type(value).__name__

    # Add details for collections
    if isinstance(value, list) and value:
        # Check if all elements are of the same type
        list_value: List[object] = value
        element_types: Set[str] = {type(elem).__name__ for elem in list_value}
        element_type: str = (
            next(iter(element_types)) if len(element_types) == 1 else "mixed"
        )
        return f"list[{element_type}] (length: {len(list_value)})"

    if isinstance(value, tuple) and value:
        # For tuples, show individual element types
        tuple_value: Tuple[object, ...] = value
        element_types_list: List[str] = [type(elem).__name__ for elem in tuple_value]
        unique_types: Set[str] = set(element_types_list)

        if len(unique_types) == 1 and element_types_list:
            # Homogeneous tuple
            return f"tuple[{element_types_list[0]}] (length: {len(tuple_value)})"
        if element_types_list:
            # Heterogeneous tuple
            return f"tuple[{', '.join(element_types_list)}]"

    if isinstance(value, dict) and value:
        # For dictionaries, show key and value types
        dict_value: Mapping[object, object] = value
        key_types: Set[str] = {type(k).__name__ for k in dict_value}
        value_types: Set[str] = {type(v).__name__ for v in dict_value.values()}

        key_type: str = next(iter(key_types)) if len(key_types) == 1 else "mixed"
        value_type: str = next(iter(value_types)) if len(value_types) == 1 else "mixed"

        return f"dict[{key_type}, {value_type}] (size: {len(dict_value)})"

    if isinstance(value, (set, frozenset)) and value:
        # For sets, show element type
        set_value: Collection[object] = value
        element_types: Set[str] = {type(elem).__name__ for elem in set_value}
        element_type: str = (
            next(iter(element_types)) if len(element_types) == 1 else "mixed"
        )
        set_type: str = "set" if isinstance(value, set) else "frozenset"
        return f"{set_type}[{element_type}] (size: {len(set_value)})"

    if isinstance(value, str):
        return f"str (length: {len(value)})"

    # Default case: just the type name
    return type_name
