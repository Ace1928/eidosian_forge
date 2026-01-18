import unittest

from type_forge.forge.type_forge import TypeForge
from type_forge.validators.basic import BasicValidator


class TestTypeForge(unittest.TestCase):

    def setUp(self):
        self.type_forge = TypeForge()
        self.validator = BasicValidator()

    def test_dynamic_type_creation(self):
        # Test dynamic type creation with a basic validator
        self.type_forge.add_validator(self.validator)
        dynamic_type = self.type_forge.create_type(
            "CustomType", {"field1": str, "field2": int}
        )
        self.assertIsNotNone(dynamic_type)
        self.assertTrue(hasattr(dynamic_type, "field1"))
        self.assertTrue(hasattr(dynamic_type, "field2"))

    def test_validation_success(self):
        # Test successful validation of a valid instance
        dynamic_type = self.type_forge.create_type(
            "ValidType", {"name": str, "age": int}
        )
        valid_instance = dynamic_type(name="Alice", age=30)
        validation_result = self.validator.validate(valid_instance)
        self.assertTrue(validation_result)

    def test_validation_failure(self):
        # Test validation failure for an invalid instance
        dynamic_type = self.type_forge.create_type(
            "InvalidType", {"name": str, "age": int}
        )
        invalid_instance = dynamic_type(name="Bob", age="not_a_number")  # Invalid age
        validation_result = self.validator.validate(invalid_instance)
        self.assertFalse(validation_result)

    def test_type_forge_integration(self):
        # Test integration of TypeForge with validators
        self.type_forge.add_validator(self.validator)
        dynamic_type = self.type_forge.create_type("IntegratedType", {"field1": str})
        instance = dynamic_type(field1="Test")
        self.assertIsInstance(instance, dynamic_type)


if __name__ == "__main__":
    unittest.main()
