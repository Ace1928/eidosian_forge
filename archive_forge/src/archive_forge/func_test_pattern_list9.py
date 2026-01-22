import unittest
from traits.api import (
def test_pattern_list9(self):
    c = Complex(tc=self)
    self.check_complex(c, c, 'int-test', ['int2'], ['int1', 'int3', 'tint4', 'tint5', 'tint6'])