import unittest
from traits.api import (
def test_pattern_list10(self):
    c = Complex(tc=self)
    self.check_complex(c, c, 'int+', ['int1', 'int2', 'int3'], ['tint1', 'tint2', 'tint3'])