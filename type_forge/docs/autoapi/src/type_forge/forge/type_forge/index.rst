.. py:module:: src.type_forge.forge.type_forge

Type Forge Core Implementation Module

   This module contains the foundational functionality for the type forging process,
   implementing dynamic type creation, validation, and transformation with recursive
   precision and structural integrity.

   The TypeForge class provides a unified interface for complex type operations:
   - Dynamic type registration and instantiation
   - Strict type validation with detailed reporting
   - Recursive schema validation for nested structures
   - Type conversion with safety guarantees

   All operations maintain full type safety through compile-time checks and
   runtime validation, ensuring system-wide integrity.


Module Contents
---------------


   .. py:class:: TypeForge   :module: 

      Core class for dynamic type creation, validation, and transformation.

      This class combines type validation, conversion, and dynamic type creation
      into a unified interface with recursive precision. It serves as the primary
      entry point for the type_forge module, providing a coherent API for all
      type-related operations.

      The system implements a recursive type validation system that can handle
      arbitrarily complex nested structures while maintaining full type safety
      and providing detailed error reporting.

      .. rubric:: Attributes

      types: Registry mapping type names to their class objects.
      validators: Collection of validators for the validation pipeline.

      .. rubric:: Examples

      >>> forge = TypeForge()
      >>> class Person: pass
      >>> forge.register_type("Person", Person)
      >>> person = forge.create_instance("Person", name="Alice", age=30)
      >>> forge.is_instance(person, "Person")
      True

      Initialize a new TypeForge instance with empty registries.

      Creates clean registries for types and validators that will be populated
      through the register_type and add_validator methods, establishing
      the foundation for dynamic type operations.



.. py:data:: __author__
      :value: 'TypeForge Team'


.. py:data:: __version__
      :value: '0.1.0'


