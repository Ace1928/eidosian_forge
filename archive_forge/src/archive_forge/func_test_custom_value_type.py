import re
import unittest
from oslo_config import types
def test_custom_value_type(self):
    self.type_instance = types.Dict(types.Integer())
    self.assertConvertedValue('foo:123, bar: 456', {'foo': 123, 'bar': 456})