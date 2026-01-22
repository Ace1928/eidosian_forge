import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
def test_from_params_dict(self):
    value = from_params(DictType(int, str), {'a[2]': 'a2', 'a[3]': 'a3'}, 'a', set())
    assert value == {2: 'a2', 3: 'a3'}, value