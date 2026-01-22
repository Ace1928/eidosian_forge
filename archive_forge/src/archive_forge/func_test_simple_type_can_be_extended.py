from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_simple_type_can_be_extended(self):

    def int_or_str_int(checker, instance):
        if not isinstance(instance, (int, str)):
            return False
        try:
            int(instance)
        except ValueError:
            return False
        return True
    CustomValidator = extend(Draft202012Validator, type_checker=Draft202012Validator.TYPE_CHECKER.redefine('integer', int_or_str_int))
    validator = CustomValidator({'type': 'integer'})
    validator.validate(4)
    validator.validate('4')
    with self.assertRaises(ValidationError):
        validator.validate(4.4)
    with self.assertRaises(ValidationError):
        validator.validate('foo')