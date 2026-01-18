type_forge.typing.hints
=======================

.. py:module:: type_forge.typing.hints

.. autoapi-nested-parse::

   Type hints for the type_forge system.

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

   .. rubric:: Attributes

   T: Generic invariant type variable for polymorphic type definitions
   K: Key type variable for dictionary-like structures
   V: Value type variable for container values



Attributes
----------

.. autoapisummary::

   type_forge.typing.hints.CollectionT
   type_forge.typing.hints.DictSchemaT
   type_forge.typing.hints.ListSchemaT
   type_forge.typing.hints.PathSegmentT
   type_forge.typing.hints.PathT
   type_forge.typing.hints.SchemaNodeT
   type_forge.typing.hints.SchemaSequenceT
   type_forge.typing.hints.SchemaTypeMappingT
   type_forge.typing.hints.SchemaTypeT
   type_forge.typing.hints.SchemaValueNodeT
   type_forge.typing.hints.SchemaValueT
   type_forge.typing.hints.SingleTypeT
   type_forge.typing.hints.UnionTypeT
   type_forge.typing.hints.ValidationResultT


Module Contents
---------------

.. py:data:: CollectionT

   Union type for common collection types with homogeneous elements.

   This type represents any of the standard Python collection types
   containing elements of the same type T, enabling generic collection operations.

   .. rubric:: Examples

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

.. py:data:: DictSchemaT

   Dictionary schema mapping string keys to schema types.

   This represents object structures with string keys mapping to schema-defined values,
   enabling the definition of complex nested objects.

   .. rubric:: Examples

   >>> person_schema: DictSchemaT = {"name": str, "age": int}
   >>> nested_schema: DictSchemaT = {"user": {"id": int, "name": str}}

.. py:data:: ListSchemaT

   List schema for array validation.

   This represents an array of items that conform to the schema type specified
   as the single element of the list.

   .. rubric:: Examples

   >>> int_list_schema: ListSchemaT = [int]
   >>> dict_list_schema: ListSchemaT = [{"name": str}]

.. py:data:: PathSegmentT

   Type representing a single segment in a path for schema traversal.

   A path segment can be either a string key (for dictionary access)
   or an integer index (for list access).

   .. rubric:: Examples

   >>> key_segment: PathSegmentT = "name"  # Access dict key
   >>> index_segment: PathSegmentT = 0     # Access list index

.. py:data:: PathT

   List of path segments representing a traversal path through a nested structure.

   Paths enable precise targeting of nested elements within complex data structures,
   combining string keys and numeric indices as needed.

   .. rubric:: Examples

   >>> # Path to access user.addresses[0].street
   >>> path: PathT = ["user", "addresses", 0, "street"]

.. py:data:: SchemaNodeT

   Generic type variable for schema nodes during traversal operations.

   This enables strongly-typed traversal of schema structures where the
   specific node type can vary based on the context.

.. py:data:: SchemaSequenceT

   Sequence representation of schema for better covariance handling.

   This provides a read-only sequence for list schemas, enabling more
   flexible handling in validation and traversal functions.

.. py:data:: SchemaTypeMappingT

   Mapping representation of schema for better covariance handling.

   This provides a read-only view of the schema structure, allowing for
   more flexible usage in functions that only need to read schema information.

.. py:data:: SchemaTypeT

   A schema type definition that can recursively represent any data structure.

   This type allows the definition of complex nested structures with full type safety,
   supporting simple types, unions of types, dictionary schemas, and list schemas.

   .. rubric:: Examples

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

.. py:data:: SchemaValueNodeT

   Generic type variable for schema value nodes during validation.

   This provides type-specific handling for values being validated against
   schema definitions, allowing for context-aware validation logic.

.. py:data:: SchemaValueT

   A value conforming to a schema definition at runtime.

   This type represents actual values that conform to a schema definition,
   allowing for runtime validation and manipulation of schema-defined structures.

   .. rubric:: Examples

   >>> # Simple value
   >>> int_value: SchemaValueT = 42
   >>>
   >>> # Dictionary value (nested object)
   >>> dict_value: SchemaValueT = {"name": "John", "age": 30}
   >>>
   >>> # List value (array)
   >>> list_value: SchemaValueT = [1, 2, 3]

.. py:data:: SingleTypeT

   Type representing a single concrete type.

   This represents any Python type that can be used for validation,
   such as built-in types (int, str) or custom classes.

   .. rubric:: Examples

   >>> int_type: SingleTypeT = int
   >>> str_type: SingleTypeT = str
   >>> custom_type: SingleTypeT = MyClass

.. py:data:: UnionTypeT

   Tuple of types representing alternatives.

   This represents a set of alternative types, similar to Union but in runtime form.

   .. rubric:: Examples

   >>> union_type: UnionTypeT = (int, str)
   >>> multi_union: UnionTypeT = (int, float, str, bool)

.. py:data:: ValidationResultT

   Generic type variable for validation results.

   This allows functions to return strongly-typed validation outcomes
   with type-specific error reporting and context.

