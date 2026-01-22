import re
import unittest
from wsme import exc
from wsme import types
def test_dynamicbase_add_attributes_second(self):

    class TempType(types.DynamicBase):
        pass
    types.registry.register(TempType)
    attrs = types.list_attributes(TempType)
    self.assertEqual(attrs, [])
    TempType.add_attributes(one=str)
    TempType.add_attributes(two=int)
    after = types.list_attributes(TempType)
    self.assertEqual(len(after), 2)