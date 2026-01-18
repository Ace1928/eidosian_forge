# EIDOSIAN VALIDATORS

The `validators` module in the `type_forge` project provides a robust framework for validating data types. This document outlines the available validators, their usage, and how to implement custom validation logic.

## Overview of Validators

Validators are essential components that ensure data integrity and correctness within the `type_forge` framework. They can be categorized into basic validators, composite validators, and custom validators.

### Basic Validators

Basic validators are designed to handle common data types and provide straightforward validation logic. They include:

- **TypeValidator**: Validates that a value is of a specified type.
- **RangeValidator**: Ensures that numeric values fall within a defined range.
- **LengthValidator**: Checks that the length of a string or collection meets specified criteria.

#### Example Usage

```python
from type_forge.validators.basic import TypeValidator, RangeValidator

# Validate an integer
int_validator = TypeValidator(int)
is_valid_int = int_validator.validate(42)  # Returns True

# Validate a number within a range
range_validator = RangeValidator(min_value=1, max_value=100)
is_valid_range = range_validator.validate(50)  # Returns True
```

### Composite Validators

Composite validators allow for the combination of multiple validation rules into a single validator. This is useful for complex data structures where multiple conditions must be met.

#### Example Usage

```python
from type_forge.validators.composite import CompositeValidator

# Create a composite validator
composite_validator = CompositeValidator([
    TypeValidator(int),
    RangeValidator(min_value=1, max_value=100)
])

is_valid_composite = composite_validator.validate(75)  # Returns True
```

### Custom Validators

For scenarios where built-in validators do not suffice, users can create custom validators by extending the base validator class. This allows for tailored validation logic specific to the application's needs.

#### Example Implementation

```python
from type_forge.validators.basic import BaseValidator

class CustomValidator(BaseValidator):
    def validate(self, value):
        # Custom validation logic
        return isinstance(value, str) and len(value) > 5

# Usage of the custom validator
custom_validator = CustomValidator()
is_valid_custom = custom_validator.validate("HelloWorld")  # Returns True
```

## Conclusion

The `validators` module in `type_forge` provides a flexible and extensible framework for data validation. By utilizing basic, composite, and custom validators, developers can ensure data integrity and enforce business rules effectively. For more advanced usage and examples, refer to the `docs/examples` section.