"""Type hints for the type_forge system.

This module defines specialized type hints and aliases that enable advanced type
checking throughout the Type Forge system. These type definitions create a strong,
statically checkable type foundation that enforces correctness at compile time
while supporting runtime introspection.

The type system supports hierarchical schemas with arbitrary nesting, allowing
for precise specification of complex data structures with full type safety.

Types defined here enable:
    - Type-safe schema validation
    - Strongly typed recursive validation
    - Generic value conversion with type preservation
    - Complex nested structure definitions with full type checking

Attributes:
    T: Generic invariant type variable for polymorphic type definitions
    K: Key type variable for dictionary-like structures
    V: Value type variable for container values
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Set, Tuple, Type, TypeVar, Union

# Type variables
T = TypeVar("T")  # Generic invariant type
K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type

# Basic schema components
SingleTypeT = Type[object]
"""Type representing a single concrete type.

This represents any Python type that can be used for validation,
such as built-in types (int, str) or custom classes.

Examples:
    >>> int_type: SingleTypeT = int
    >>> str_type: SingleTypeT = str
    >>> custom_type: SingleTypeT = MyClass
"""

UnionTypeT = Tuple[Type[object], ...]
"""Tuple of types representing alternatives.

This represents a set of alternative types, similar to Union but in runtime form.

Examples:
    >>> union_type: UnionTypeT = (int, str)
    >>> multi_union: UnionTypeT = (int, float, str, bool)
"""

# Forward references for recursive type definitions
SchemaTypeT = Union[
    SingleTypeT,  # Simple type (int, str, etc.)
    UnionTypeT,  # Tuple of types (like Union)
    Dict[str, "SchemaTypeT"],  # Dict schema (nested object) with string keys
    List["SchemaTypeT"],  # List schema (array) with single schema definition
]
"""A schema type definition that can recursively represent any data structure.

This type allows the definition of complex nested structures with full type safety,
supporting simple types, unions of types, dictionary schemas, and list schemas.

Examples:
    >>> # Simple type schema
    >>> int_schema: SchemaTypeT = int
    >>>
    >>> # Union type schema (alternative types)
    >>> union_schema: SchemaTypeT = (int, str)
    >>>
    >>> # Dictionary schema (nested object)
    >>> dict_schema: SchemaTypeT = {"name": str, "age": int}
    >>>
    >>> # List schema (array of integers)
    >>> list_schema: SchemaTypeT = [int]
"""

# Schema-specific container types with proper typing
DictSchemaT = Dict[str, SchemaTypeT]
"""Dictionary schema mapping string keys to schema types.

This represents object structures with string keys mapping to schema-defined values,
enabling the definition of complex nested objects.

Examples:
    >>> person_schema: DictSchemaT = {"name": str, "age": int}
    >>> nested_schema: DictSchemaT = {"user": {"id": int, "name": str}}
"""

ListSchemaT = List[SchemaTypeT]
"""List schema for array validation.

This represents an array of items that conform to the schema type specified
as the single element of the list.

Examples:
    >>> int_list_schema: ListSchemaT = [int]
    >>> dict_list_schema: ListSchemaT = [{"name": str}]
"""

# Schema value type for runtime representation
SchemaValueT = Union[
    object,  # Simple value
    Dict[str, "SchemaValueT"],  # Dict value (nested object)
    List["SchemaValueT"],  # List value (array)
]
"""A value conforming to a schema definition at runtime.

This type represents actual values that conform to a schema definition,
allowing for runtime validation and manipulation of schema-defined structures.

Examples:
    >>> # Simple value
    >>> int_value: SchemaValueT = 42
    >>>
    >>> # Dictionary value (nested object)
    >>> dict_value: SchemaValueT = {"name": "John", "age": 30}
    >>>
    >>> # List value (array)
    >>> list_value: SchemaValueT = [1, 2, 3]
"""

# For better covariance handling in advanced use cases
SchemaTypeMappingT = Mapping[str, SchemaTypeT]
"""Mapping representation of schema for better covariance handling.

This provides a read-only view of the schema structure, allowing for
more flexible usage in functions that only need to read schema information.
"""

SchemaSequenceT = Sequence[SchemaTypeT]
"""Sequence representation of schema for better covariance handling.

This provides a read-only sequence for list schemas, enabling more
flexible handling in validation and traversal functions.
"""

# Results and collections
ValidationResultT = TypeVar("ValidationResultT")
"""Generic type variable for validation results.

This allows functions to return strongly-typed validation outcomes
with type-specific error reporting and context.
"""

CollectionT = Union[List[T], Dict[str, T], Tuple[T, ...], Set[T]]
"""Union type for common collection types with homogeneous elements.

This type represents any of the standard Python collection types
containing elements of the same type T, enabling generic collection operations.

Examples:
    >>> # List collection
    >>> names: CollectionT[str] = ["Alice", "Bob", "Charlie"]
    >>>
    >>> # Dictionary collection
    >>> ages: CollectionT[int] = {"Alice": 30, "Bob": 25, "Charlie": 35}
    >>>
    >>> # Tuple collection
    >>> coordinates: CollectionT[float] = (1.0, 2.0, 3.0)
    >>>
    >>> # Set collection
    >>> unique_ids: CollectionT[int] = {1, 2, 3}
"""

# Path representation for schema traversal
PathSegmentT = Union[str, int]
"""Type representing a single segment in a path for schema traversal.

A path segment can be either a string key (for dictionary access)
or an integer index (for list access).

Examples:
    >>> key_segment: PathSegmentT = "name"  # Access dict key
    >>> index_segment: PathSegmentT = 0     # Access list index
"""

PathT = List[PathSegmentT]
"""List of path segments representing a traversal path through a nested structure.

Paths enable precise targeting of nested elements within complex data structures,
combining string keys and numeric indices as needed.

Examples:
    >>> # Path to access user.addresses[0].street
    >>> path: PathT = ["user", "addresses", 0, "street"]
"""

# Schema traversal types
SchemaNodeT = TypeVar("SchemaNodeT")
"""Generic type variable for schema nodes during traversal operations.

This enables strongly-typed traversal of schema structures where the
specific node type can vary based on the context.
"""

SchemaValueNodeT = TypeVar("SchemaValueNodeT")
"""Generic type variable for schema value nodes during validation.

This provides type-specific handling for values being validated against
schema definitions, allowing for context-aware validation logic.
"""

__all__ = [
    "SchemaTypeT",
    "SchemaValueT",
    "DictSchemaT",
    "SingleTypeT",
    "UnionTypeT",
    "ListSchemaT",
    "SchemaTypeMappingT",
    "SchemaSequenceT",
    "ValidationResultT",
    "CollectionT",
    "PathSegmentT",
    "PathT",
    "SchemaNodeT",
    "SchemaValueNodeT",
]
