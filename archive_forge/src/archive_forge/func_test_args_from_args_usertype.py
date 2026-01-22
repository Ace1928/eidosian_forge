import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
def test_args_from_args_usertype(self):

    class FakeType(UserType):
        name = 'fake-type'
        basetype = int
    fake_type = FakeType()
    fd = FunctionDefinition(FunctionDefinition)
    fd.arguments.append(FunctionArgument('fake-arg', fake_type, True, 0))
    new_args = args_from_args(fd, [1], {})
    self.assertEqual([1], new_args[0])
    try:
        args_from_args(fd, ['invalid-argument'], {})
    except InvalidInput as e:
        assert fake_type.name in str(e)
    else:
        self.fail('Should have thrown an InvalidInput')