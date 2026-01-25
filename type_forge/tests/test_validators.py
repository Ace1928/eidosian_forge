"""Tests for TypeCore schema constraints validation."""
import unittest
from type_forge import TypeCore, ValidationError


class TestConstraintValidation(unittest.TestCase):
    """Test schema constraint validations."""
    
    def setUp(self):
        self.core = TypeCore()

    def test_validate_integer(self):
        """Test integer type validation."""
        schema = {"type": "integer"}
        self.assertTrue(self.core.check_schema(schema, 10))
        
        with self.assertRaises(ValidationError):
            self.core.check_schema(schema, "10")

    def test_validate_string(self):
        """Test string type validation."""
        schema = {"type": "string"}
        self.assertTrue(self.core.check_schema(schema, "hello"))
        
        with self.assertRaises(ValidationError):
            self.core.check_schema(schema, 10)

    def test_number_constraints(self):
        """Test number range constraints."""
        schema = {"type": "number", "minimum": 0, "maximum": 100}
        
        self.assertTrue(self.core.check_schema(schema, 50))
        self.assertTrue(self.core.check_schema(schema, 0))
        self.assertTrue(self.core.check_schema(schema, 100))
        
        with self.assertRaises(ValidationError):
            self.core.check_schema(schema, -1)
        with self.assertRaises(ValidationError):
            self.core.check_schema(schema, 101)

    def test_string_constraints(self):
        """Test string length constraints."""
        schema = {"type": "string", "minLength": 3, "maxLength": 10}
        
        self.assertTrue(self.core.check_schema(schema, "abc"))
        self.assertTrue(self.core.check_schema(schema, "0123456789"))
        
        with self.assertRaises(ValidationError):
            self.core.check_schema(schema, "ab")
        with self.assertRaises(ValidationError):
            self.core.check_schema(schema, "01234567890")


if __name__ == '__main__':
    unittest.main()