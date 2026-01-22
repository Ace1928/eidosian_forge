from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def int_or_str_int(checker, instance):
    if not isinstance(instance, (int, str)):
        return False
    try:
        int(instance)
    except ValueError:
        return False
    return True