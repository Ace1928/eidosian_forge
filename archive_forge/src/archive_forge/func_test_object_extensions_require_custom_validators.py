from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_object_extensions_require_custom_validators(self):
    schema = {'type': 'object', 'required': ['x']}
    type_checker = Draft202012Validator.TYPE_CHECKER.redefine('object', is_object_or_named_tuple)
    CustomValidator = extend(Draft202012Validator, type_checker=type_checker)
    validator = CustomValidator(schema)
    Point = namedtuple('Point', ['x', 'y'])
    with self.assertRaises(ValidationError):
        validator.validate(Point(x=4, y=5))