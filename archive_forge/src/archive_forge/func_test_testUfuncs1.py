from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testUfuncs1(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf, s = self.d
    assert_(eq(np.cos(x), cos(xm)))
    assert_(eq(np.cosh(x), cosh(xm)))
    assert_(eq(np.sin(x), sin(xm)))
    assert_(eq(np.sinh(x), sinh(xm)))
    assert_(eq(np.tan(x), tan(xm)))
    assert_(eq(np.tanh(x), tanh(xm)))
    with np.errstate(divide='ignore', invalid='ignore'):
        assert_(eq(np.sqrt(abs(x)), sqrt(xm)))
        assert_(eq(np.log(abs(x)), log(xm)))
        assert_(eq(np.log10(abs(x)), log10(xm)))
    assert_(eq(np.exp(x), exp(xm)))
    assert_(eq(np.arcsin(z), arcsin(zm)))
    assert_(eq(np.arccos(z), arccos(zm)))
    assert_(eq(np.arctan(z), arctan(zm)))
    assert_(eq(np.arctan2(x, y), arctan2(xm, ym)))
    assert_(eq(np.absolute(x), absolute(xm)))
    assert_(eq(np.equal(x, y), equal(xm, ym)))
    assert_(eq(np.not_equal(x, y), not_equal(xm, ym)))
    assert_(eq(np.less(x, y), less(xm, ym)))
    assert_(eq(np.greater(x, y), greater(xm, ym)))
    assert_(eq(np.less_equal(x, y), less_equal(xm, ym)))
    assert_(eq(np.greater_equal(x, y), greater_equal(xm, ym)))
    assert_(eq(np.conjugate(x), conjugate(xm)))
    assert_(eq(np.concatenate((x, y)), concatenate((xm, ym))))
    assert_(eq(np.concatenate((x, y)), concatenate((x, y))))
    assert_(eq(np.concatenate((x, y)), concatenate((xm, y))))
    assert_(eq(np.concatenate((x, y, x)), concatenate((x, ym, x))))