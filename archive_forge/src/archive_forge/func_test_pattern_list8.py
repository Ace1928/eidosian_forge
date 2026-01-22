import unittest
from traits.api import (
def test_pattern_list8(self):
    c = Complex(tc=self)
    self.check_complex(c, c, 'int+test', ['int1', 'int3'], ['int2', 'tint1', 'tint2', 'tint3'])