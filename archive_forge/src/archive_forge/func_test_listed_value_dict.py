import re
import unittest
from oslo_config import types
def test_listed_value_dict(self):
    self.type_instance = types.String(choices=[('foo', 'ab'), ('bar', 'xy')])
    self.assertConvertedValue('foo', 'foo')