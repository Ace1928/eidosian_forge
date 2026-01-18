import unittest
from type_forge.core.base import BaseValidator
from type_forge.core.exceptions import ValidationError

class TestBaseValidator(unittest.TestCase):
    
    def setUp(self):
        self.validator = BaseValidator()

    def test_valid_data(self):
        valid_data = "test_string"
        self.assertTrue(self.validator.validate(valid_data))

    def test_invalid_data(self):
        invalid_data = 12345
        with self.assertRaises(ValidationError):
            self.validator.validate(invalid_data)

    def test_empty_data(self):
        empty_data = ""
        with self.assertRaises(ValidationError):
            self.validator.validate(empty_data)

    def test_type_check(self):
        self.assertTrue(self.validator.is_valid_type("string"))
        self.assertFalse(self.validator.is_valid_type(123))

if __name__ == '__main__':
    unittest.main()