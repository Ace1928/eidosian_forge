# EIDOSIAN CUSTOM VALIDATORS EXAMPLE

from type_forge.validators.basic import StringValidator, IntegerValidator
from type_forge.validators.composite import CompositeValidator

class CustomStringValidator(StringValidator):
    def validate(self, value: str) -> bool:
        super().validate(value)
        return len(value) > 5  # Custom rule: string must be longer than 5 characters

class CustomIntegerValidator(IntegerValidator):
    def validate(self, value: int) -> bool:
        super().validate(value)
        return value % 2 == 0  # Custom rule: integer must be even

# Example usage of custom validators
def main():
    string_validator = CustomStringValidator()
    integer_validator = CustomIntegerValidator()

    test_strings = ["hello", "world!", "hi"]
    test_integers = [2, 3, 4]

    for test_string in test_strings:
        if string_validator.validate(test_string):
            print(f"'{test_string}' is a valid string.")
        else:
            print(f"'{test_string}' is invalid.")

    for test_integer in test_integers:
        if integer_validator.validate(test_integer):
            print(f"{test_integer} is a valid integer.")
        else:
            print(f"{test_integer} is invalid.")

if __name__ == "__main__":
    main()