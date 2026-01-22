from numba import int32, int64, uint32, uint64, float32, float64
from numba.core.types import range_iter32_type
from numba.core import itanium_mangler
import unittest
def test_mangle_literal(self):
    got = itanium_mangler.mangle_value(123)
    expect = 'Li123E'
    self.assertEqual(expect, got)
    got = itanium_mangler.mangle_value(12.3)
    self.assertRegex(got, '^\\d+_12_[0-9a-z][0-9a-z]3$')