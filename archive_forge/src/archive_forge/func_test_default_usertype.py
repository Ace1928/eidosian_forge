import re
import unittest
from wsme import exc
from wsme import types
def test_default_usertype(self):

    class MyType(types.UserType):
        basetype = str
    My = MyType()
    assert My.validate('a') == 'a'
    assert My.tobasetype('a') == 'a'
    assert My.frombasetype('a') == 'a'