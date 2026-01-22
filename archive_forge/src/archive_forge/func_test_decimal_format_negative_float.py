import re
import unittest
from oslo_config import types
def test_decimal_format_negative_float(self):
    v = self.type_instance('-123.456')
    self.assertAlmostEqual(v, -123.456)