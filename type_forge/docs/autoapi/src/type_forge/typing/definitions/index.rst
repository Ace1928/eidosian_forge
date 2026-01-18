.. py:module:: src.type_forge.typing.definitions

Type Definitions for the Type Forge system.


Module Contents
---------------


   .. py:class:: TypeCategory(*args, **kwds)   :module: 

      Categorization of type structures for semantic operations.

      This enumeration provides a richer vocabulary for describing type
      relationships beyond basic inheritance, allowing for precise categorization
      of types based on their structural and behavioral properties.

      .. rubric:: Attributes

      ATOMIC: Primitive types that are indivisible units (int, float, bool)
      COMPOSITE: Types composed of other types (classes, dataclasses)
      STRUCTURAL: Types that define a structure or shape (NamedTuple, Struct)
      CONTAINER: Types that hold collections of other values (List, Dict)
      FUNCTION: Types representing callable objects (functions, lambdas)
      PROTOCOL: Types representing interfaces or behaviors (Protocols)
      GENERIC: Types with type parameters (List[T], Dict[K, V])
      SPECIAL: Types with special behaviors (Optional, Union)
      NUMERIC: Types representing numbers (int, float, complex)
      TEXT: Types representing text (str, bytes)
      BOOLEAN: Types representing boolean values (bool)
      DATETIME: Types representing dates and times
      ENUM: Enumeration types
      BINARY: Binary data types (bytes, bytearray)
      NETWORK: Network-related types (URL, IP address)
      RECURSIVE: Self-referential types

      .. rubric:: Examples

      >>> TypeCategory.ATOMIC.name
      'ATOMIC'
      >>> TypeCategory.CONTAINER.value
      'container'
      >>> str(TypeCategory.FUNCTION)
      'function'




   .. py:class:: TypeCompatibility(*args, **kwds)   :module: 

      Classification of type compatibility relationships for conversion operations.

      This enumeration provides a detailed classification system for describing
      the compatibility between two types, guiding conversion strategies and
      validation processes.

      .. rubric:: Attributes

      IDENTICAL: Types are exactly the same (int and int).
      SUBTYPE: Source is a subtype of target (bool is a subtype of int).
      SUPERTYPE: Target is a subtype of source (int is a supertype of bool).
      CONVERTIBLE: Types can be converted explicitly (str to int).
      STRUCTURALLY_COMPATIBLE: Types share compatible structures (NamedTuple vs dataclass).
      PROTOCOL_COMPATIBLE: Source satisfies target's protocol requirements.
      CONTAINER_COMPATIBLE: Container types with compatible element types.
      IMPLICIT_CONVERTIBLE: Types can be converted implicitly (int to float).
      INCOMPATIBLE: Types cannot be converted or are fundamentally different.

      .. rubric:: Examples

      >>> TypeCompatibility.IDENTICAL.name
      'IDENTICAL'
      >>> TypeCompatibility.CONVERTIBLE.value
      'convertible'
      >>> str(TypeCompatibility.INCOMPATIBLE)
      'incompatible'




   .. py:class:: ValidationLevel(*args, **kwds)   :module: 

      Levels of validation strictness for type validation functions.

      This enumeration provides different levels of strictness for
      type validation operations, allowing for flexible type checking
      based on application requirements.

      .. rubric:: Attributes

      STRICT: Allow no type variance, exact type match required.
          Types must be identical (A is A, not A is subclass of B).
      STANDARD: Allow normal subtype relationships (inheritance).
          Types can be subclasses (B is acceptable when A is required if B inherits
          from A).
      PERMISSIVE: Allow type conversion where possible.
          Attempts to convert between compatible types (str to int if str contains
          a number).
      DYNAMIC: Use duck typing and runtime checks.
          Checks for attribute/method presence rather than type identity.
      STRUCTURAL: Check structural compatibility only.
          Types are compatible if they have compatible structures regardless
          of inheritance.
      COVARIANT: Allow covariant substitution.
          A type B can be used where A is required if B is a subtype of A.
      CONTRAVARIANT: Allow contravariant substitution.
          A type B can be used where A is required if A is a subtype of B.
      NONE: No validation performed.
          All types are accepted without verification (use with caution).

      .. rubric:: Examples

      >>> ValidationLevel.STRICT.name
      'STRICT'
      >>> ValidationLevel.STANDARD.value
      'standard'
      >>> str(ValidationLevel.PERMISSIVE)
      'permissive'




   .. py:class:: ValidationSeverity(*args, **kwds)   :module: 

      Severity levels for validation errors and warnings.

      This enumeration provides different severity levels for validation
      issues, enabling appropriate handling and reporting based on criticality.

      .. rubric:: Attributes

      FATAL: Critical error that must be fixed and prevents operation.
      ERROR: Serious issue that should be fixed but might allow partial operation.
      WARNING: Potential issue or deviation from best practice.
      INFO: Informational message about validation results.
      DEBUG: Technical details useful for debugging validation.

      .. rubric:: Examples

      >>> ValidationSeverity.ERROR.name
      'ERROR'
      >>> ValidationSeverity.WARNING.value
      'warning'
      >>> str(ValidationSeverity.INFO)
      'info'



.. py:data:: version
      :value: '0.1.0'


