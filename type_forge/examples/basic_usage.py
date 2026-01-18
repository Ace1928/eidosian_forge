# examples/basic_usage.py

from type_forge.forge.type_forge import TypeForge
from type_forge.validators.basic import IntegerValidator, StringValidator


def main():
    # Initialize the TypeForge instance
    forge = TypeForge()

    # Define some sample data
    sample_data = {"name": "Alice", "age": 30, "email": "alice@example.com"}

    # Define validation rules
    validators = {
        "name": StringValidator(max_length=50),
        "age": IntegerValidator(min_value=0, max_value=120),
        "email": StringValidator(max_length=100),
    }

    # Validate the sample data
    for field, validator in validators.items():
        value = sample_data[field]
        if validator.validate(value):
            print(f"{field} is valid: {value}")
        else:
            print(f"{field} is invalid: {value}")


if __name__ == "__main__":
    main()
