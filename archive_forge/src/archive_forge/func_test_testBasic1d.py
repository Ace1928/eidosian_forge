from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testBasic1d(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf, s = self.d
    assert_(not isMaskedArray(x))
    assert_(isMaskedArray(xm))
    assert_equal(shape(xm), s)
    assert_equal(xm.shape, s)
    assert_equal(xm.dtype, x.dtype)
    assert_equal(xm.size, reduce(lambda x, y: x * y, s))
    assert_equal(count(xm), len(m1) - reduce(lambda x, y: x + y, m1))
    assert_(eq(xm, xf))
    assert_(eq(filled(xm, 1e+20), xf))
    assert_(eq(x, xm))