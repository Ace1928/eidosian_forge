from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_is_type(self):
    checker = TypeChecker({'two': equals_2})
    self.assertEqual((checker.is_type(instance=2, type='two'), checker.is_type(instance='bar', type='two')), (True, False))