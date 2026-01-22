from numba import int32, int64, uint32, uint64, float32, float64
from numba.core.types import range_iter32_type
from numba.core import itanium_mangler
import unittest
def test_custom_type(self):
    got = itanium_mangler.mangle_type(range_iter32_type)
    name = str(range_iter32_type)
    expect = '{n}{name}'.format(n=len(name), name=name)
    self.assertEqual(expect, got)