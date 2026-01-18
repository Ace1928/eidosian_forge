# Basic Usage of the type_forge Module

This document provides examples of how to use the `type_forge` module effectively. The examples demonstrate the core functionalities and features of the module, allowing users to understand its capabilities and how to implement them in their own projects.

## Example 1: Basic Type Forging

In this example, we will create a simple type using the `type_forge` module. This will illustrate how to define a type and validate its instances.

```python
from type_forge.forge import TypeForge

# Define a simple type
UserType = TypeForge.create_type(
    name="User",
    fields={
        "username": str,
        "age": int
    }
)

# Create an instance of the User type
user_instance = UserType(username="john_doe", age=30)

# Validate the instance
if user_instance.is_valid():
    print("User instance is valid:", user_instance)
else:
    print("User instance is invalid:", user_instance.errors)
```

## Example 2: Using Validators

This example demonstrates how to use built-in validators to enforce constraints on the data types.

```python
from type_forge.validators.basic import StringValidator, IntegerValidator

# Create validators
username_validator = StringValidator(min_length=3, max_length=20)
age_validator = IntegerValidator(min_value=0, max_value=120)

# Validate data
username = "john_doe"
age = 30

if username_validator.validate(username) and age_validator.validate(age):
    print("Both username and age are valid.")
else:
    print("Validation failed.")
```

## Example 3: Custom Type Creation

In this example, we will create a custom type with specific validation rules.

```python
from type_forge.forge import TypeForge
from type_forge.validators.composite import CompositeValidator

# Define a custom validator
class UserValidator(CompositeValidator):
    def validate(self, user):
        return (
            username_validator.validate(user.username) and
            age_validator.validate(user.age)
        )

# Create a custom type with the validator
CustomUserType = TypeForge.create_type(
    name="CustomUser",
    fields={
        "username": str,
        "age": int
    },
    validator=UserValidator()
)

# Create an instance of the CustomUser type
custom_user_instance = CustomUserType(username="john_doe", age=30)

# Validate the instance
if custom_user_instance.is_valid():
    print("Custom user instance is valid:", custom_user_instance)
else:
    print("Custom user instance is invalid:", custom_user_instance.errors)
```

## Conclusion

The `type_forge` module provides a powerful framework for creating and validating types in Python. By following the examples above, users can quickly get started with defining their own types and implementing validation logic tailored to their needs. For more advanced usage and features, please refer to the documentation in the `docs` directory.