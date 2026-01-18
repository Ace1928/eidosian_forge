from type_forge import BasicValidator, TypeForge

# Create a type forge instance
forge = TypeForge()

# Define a simple person type with validation
Person = forge.create_type(
    name="Person",
    fields={"name": str, "age": int, "email": str},
    validators={
        "age": BasicValidator().validate_integer(min_value=0, max_value=120),
        "email": BasicValidator().validate_string(pattern=r".*@.*\..*"),
    },
)

# Create a person instance
person = Person(name="John Doe", age=30, email="john@example.com")

# This will raise a validation error
try:
    invalid_person = Person(name="Invalid", age=-5, email="not-an-email")
except ValueError as e:
    print(f"Validation error: {e}")
