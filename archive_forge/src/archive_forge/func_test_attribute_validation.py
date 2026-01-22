import re
import unittest
from wsme import exc
from wsme import types
def test_attribute_validation(self):

    class AType(object):
        alist = [int]
        aint = int
    types.register_type(AType)
    obj = AType()
    obj.alist = [1, 2, 3]
    assert obj.alist == [1, 2, 3]
    obj.aint = 5
    assert obj.aint == 5
    self.assertRaises(exc.InvalidInput, setattr, obj, 'alist', 12)
    self.assertRaises(exc.InvalidInput, setattr, obj, 'alist', [2, 'a'])