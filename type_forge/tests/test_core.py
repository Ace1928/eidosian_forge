"""Tests for TypeCore validation functionality."""
import unittest
from type_forge import TypeCore, ValidationError

class TestValidation(unittest.TestCase):
    """Test TypeCore validation methods."""
    
    def setUp(self):
        self.core = TypeCore()

    def test_string_validation(self):
        """Test basic string type validation."""
        schema = {"type": "string"}
        self.assertTrue(self.core.check_schema(schema, "test_string"))
        
        with self.assertRaises(ValidationError):
            self.core.check_schema(schema, 12345)

    def test_empty_string(self):
        """Test empty strings pass basic string validation."""
        schema = {"type": "string"}
        # Empty string is still a valid string
        self.assertTrue(self.core.check_schema(schema, ""))

    def test_string_min_length(self):
        """Test string minLength validation."""
        schema = {"type": "string", "minLength": 1}
        with self.assertRaises(ValidationError):
            self.core.check_schema(schema, "")
        self.assertTrue(self.core.check_schema(schema, "a"))

    def test_type_validation(self):
        """Test various type validations."""
        self.assertTrue(self.core.check_schema({"type": "string"}, "test"))
        self.assertTrue(self.core.check_schema({"type": "number"}, 123.5))
        self.assertTrue(self.core.check_schema({"type": "integer"}, 123))
        self.assertTrue(self.core.check_schema({"type": "boolean"}, True))

if __name__ == '__main__':
    unittest.main()