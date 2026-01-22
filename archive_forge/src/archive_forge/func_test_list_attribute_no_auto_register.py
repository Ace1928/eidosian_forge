import re
import unittest
from wsme import exc
from wsme import types
def test_list_attribute_no_auto_register(self):

    class MyType(object):
        aint = int
    assert not hasattr(MyType, '_wsme_attributes')
    self.assertRaises(TypeError, types.list_attributes, MyType)
    assert not hasattr(MyType, '_wsme_attributes')