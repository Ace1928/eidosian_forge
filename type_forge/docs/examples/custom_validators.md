# Custom Validators in Type Forge

This document provides examples of how to create and use custom validators within the Type Forge module. Custom validators allow you to define specific validation rules tailored to your application's needs, enhancing the flexibility and robustness of the type validation framework.

## Creating a Custom Validator

To create a custom validator, you need to define a class that implements the validation logic. Below is an example of a simple custom validator that checks if a value is an even number.

```python
from type_forge.validators.base import BaseValidator

class EvenNumberValidator(BaseValidator):
    def validate(self, value: int) -> bool:
        """Validate that the number is even."""
        if not isinstance(value, int):
            raise ValueError("Value must be an integer.")
        return value % 2 == 0
```

### Usage of the Custom Validator

Once you have defined your custom validator, you can use it in your type validation process. Here’s how to apply the `EvenNumberValidator`:

```python
from type_forge.forge import TypeForge

# Create an instance of the TypeForge
forge = TypeForge()

# Register the custom validator
forge.register_validator(EvenNumberValidator())

# Validate a value
try:
    result = forge.validate(4, validator=EvenNumberValidator())
    print("Validation passed:", result)
except ValueError as e:
    print("Validation failed:", e)
```

## Combining Custom Validators

You can also combine multiple custom validators to create more complex validation rules. Here’s an example of a composite validator that checks if a number is both even and positive.

```python
from type_forge.validators.composite import CompositeValidator

class EvenAndPositiveValidator(CompositeValidator):
    def __init__(self):
        super().__init__([
            EvenNumberValidator(),
            PositiveNumberValidator()  # Assume this is defined elsewhere
        ])

# Usage
forge.register_validator(EvenAndPositiveValidator())
```

## Conclusion

Custom validators in Type Forge provide a powerful way to enforce specific validation rules in your applications. By extending the base validator class, you can create tailored solutions that fit your unique requirements, ensuring that your data adheres to the expected formats and constraints.