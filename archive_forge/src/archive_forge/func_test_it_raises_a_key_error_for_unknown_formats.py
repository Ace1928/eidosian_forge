from unittest import TestCase
from jsonschema import FormatChecker, ValidationError
from jsonschema.exceptions import FormatError
from jsonschema.validators import Draft4Validator
def test_it_raises_a_key_error_for_unknown_formats(self):
    with self.assertRaises(KeyError):
        FormatChecker(formats=['o noes'])