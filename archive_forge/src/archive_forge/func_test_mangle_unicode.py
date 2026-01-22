from numba import int32, int64, uint32, uint64, float32, float64
from numba.core.types import range_iter32_type
from numba.core import itanium_mangler
import unittest
def test_mangle_unicode(self):
    name = u'f∂ƒ©z'
    got = itanium_mangler.mangle_identifier(name)
    self.assertRegex(got, '^\\d+f(_[a-z0-9][a-z0-9])+z$')