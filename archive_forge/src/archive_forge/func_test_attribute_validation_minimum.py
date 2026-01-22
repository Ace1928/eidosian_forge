import re
import unittest
from wsme import exc
from wsme import types
def test_attribute_validation_minimum(self):

    class ATypeInt(object):
        attr = types.IntegerType(minimum=1, maximum=5)
    types.register_type(ATypeInt)
    obj = ATypeInt()
    obj.attr = 2
    self.assertRaises(exc.InvalidInput, setattr, obj, 'attr', 'zero')