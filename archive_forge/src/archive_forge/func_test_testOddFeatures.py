from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testOddFeatures(self):
    x = arange(20)
    x = x.reshape(4, 5)
    x.flat[5] = 12
    assert_(x[1, 0] == 12)
    z = x + 10j * x
    assert_(eq(z.real, x))
    assert_(eq(z.imag, 10 * x))
    assert_(eq((z * conjugate(z)).real, 101 * x * x))
    z.imag[...] = 0.0
    x = arange(10)
    x[3] = masked
    assert_(str(x[3]) == str(masked))
    c = x >= 8
    assert_(count(where(c, masked, masked)) == 0)
    assert_(shape(where(c, masked, masked)) == c.shape)
    z = where(c, x, masked)
    assert_(z.dtype is x.dtype)
    assert_(z[3] is masked)
    assert_(z[4] is masked)
    assert_(z[7] is masked)
    assert_(z[8] is not masked)
    assert_(z[9] is not masked)
    assert_(eq(x, z))
    z = where(c, masked, x)
    assert_(z.dtype is x.dtype)
    assert_(z[3] is masked)
    assert_(z[4] is not masked)
    assert_(z[7] is not masked)
    assert_(z[8] is masked)
    assert_(z[9] is masked)
    z = masked_where(c, x)
    assert_(z.dtype is x.dtype)
    assert_(z[3] is masked)
    assert_(z[4] is not masked)
    assert_(z[7] is not masked)
    assert_(z[8] is masked)
    assert_(z[9] is masked)
    assert_(eq(x, z))
    x = array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = array([1, 1, 1, 0, 0])
    x[2] = masked
    z = where(c, x, -x)
    assert_(eq(z, [1.0, 2.0, 0.0, -4.0, -5]))
    c[0] = masked
    z = where(c, x, -x)
    assert_(eq(z, [1.0, 2.0, 0.0, -4.0, -5]))
    assert_(z[0] is masked)
    assert_(z[1] is not masked)
    assert_(z[2] is masked)
    assert_(eq(masked_where(greater(x, 2), x), masked_greater(x, 2)))
    assert_(eq(masked_where(greater_equal(x, 2), x), masked_greater_equal(x, 2)))
    assert_(eq(masked_where(less(x, 2), x), masked_less(x, 2)))
    assert_(eq(masked_where(less_equal(x, 2), x), masked_less_equal(x, 2)))
    assert_(eq(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2)))
    assert_(eq(masked_where(equal(x, 2), x), masked_equal(x, 2)))
    assert_(eq(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2)))
    assert_(eq(masked_inside(list(range(5)), 1, 3), [0, 199, 199, 199, 4]))
    assert_(eq(masked_outside(list(range(5)), 1, 3), [199, 1, 2, 3, 199]))
    assert_(eq(masked_inside(array(list(range(5)), mask=[1, 0, 0, 0, 0]), 1, 3).mask, [1, 1, 1, 1, 0]))
    assert_(eq(masked_outside(array(list(range(5)), mask=[0, 1, 0, 0, 0]), 1, 3).mask, [1, 1, 0, 0, 1]))
    assert_(eq(masked_equal(array(list(range(5)), mask=[1, 0, 0, 0, 0]), 2).mask, [1, 0, 1, 0, 0]))
    assert_(eq(masked_not_equal(array([2, 2, 1, 2, 1], mask=[1, 0, 0, 0, 0]), 2).mask, [1, 0, 1, 0, 1]))
    assert_(eq(masked_where([1, 1, 0, 0, 0], [1, 2, 3, 4, 5]), [99, 99, 3, 4, 5]))
    atest = ones((10, 10, 10), dtype=np.float32)
    btest = zeros(atest.shape, MaskType)
    ctest = masked_where(btest, atest)
    assert_(eq(atest, ctest))
    z = choose(c, (-x, x))
    assert_(eq(z, [1.0, 2.0, 0.0, -4.0, -5]))
    assert_(z[0] is masked)
    assert_(z[1] is not masked)
    assert_(z[2] is masked)
    x = arange(6)
    x[5] = masked
    y = arange(6) * 10
    y[2] = masked
    c = array([1, 1, 1, 0, 0, 0], mask=[1, 0, 0, 0, 0, 0])
    cm = c.filled(1)
    z = where(c, x, y)
    zm = where(cm, x, y)
    assert_(eq(z, zm))
    assert_(getmask(zm) is nomask)
    assert_(eq(zm, [0, 1, 2, 30, 40, 50]))
    z = where(c, masked, 1)
    assert_(eq(z, [99, 99, 99, 1, 1, 1]))
    z = where(c, 1, masked)
    assert_(eq(z, [99, 1, 1, 99, 99, 99]))