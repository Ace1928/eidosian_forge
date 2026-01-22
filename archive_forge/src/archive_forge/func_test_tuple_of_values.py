import re
import unittest
from oslo_config import types
def test_tuple_of_values(self):
    self.assertConvertedValue(('foo', 'bar'), ['foo', 'bar'])