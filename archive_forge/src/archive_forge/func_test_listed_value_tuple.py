import re
import unittest
from oslo_config import types
def test_listed_value_tuple(self):
    self.type_instance = types.String(choices=('foo', 'bar'))
    self.assertConvertedValue('foo', 'foo')