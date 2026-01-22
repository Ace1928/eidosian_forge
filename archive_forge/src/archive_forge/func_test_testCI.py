from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testCI(self):
    x1 = np.array([1, 2, 4, 3])
    x2 = array(x1, mask=[1, 0, 0, 0])
    x3 = array(x1, mask=[0, 1, 0, 1])
    x4 = array(x1)
    str(x2)
    repr(x2)
    assert_(eq(np.sort(x1), sort(x2, fill_value=0)))
    assert_(type(x2[1]) is type(x1[1]))
    assert_(x1[1] == x2[1])
    assert_(x2[0] is masked)
    assert_(eq(x1[2], x2[2]))
    assert_(eq(x1[2:5], x2[2:5]))
    assert_(eq(x1[:], x2[:]))
    assert_(eq(x1[1:], x3[1:]))
    x1[2] = 9
    x2[2] = 9
    assert_(eq(x1, x2))
    x1[1:3] = 99
    x2[1:3] = 99
    assert_(eq(x1, x2))
    x2[1] = masked
    assert_(eq(x1, x2))
    x2[1:3] = masked
    assert_(eq(x1, x2))
    x2[:] = x1
    x2[1] = masked
    assert_(allequal(getmask(x2), array([0, 1, 0, 0])))
    x3[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
    assert_(allequal(getmask(x3), array([0, 1, 1, 0])))
    x4[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
    assert_(allequal(getmask(x4), array([0, 1, 1, 0])))
    assert_(allequal(x4, array([1, 2, 3, 4])))
    x1 = np.arange(5) * 1.0
    x2 = masked_values(x1, 3.0)
    assert_(eq(x1, x2))
    assert_(allequal(array([0, 0, 0, 1, 0], MaskType), x2.mask))
    assert_(eq(3.0, x2.fill_value))
    x1 = array([1, 'hello', 2, 3], object)
    x2 = np.array([1, 'hello', 2, 3], object)
    s1 = x1[1]
    s2 = x2[1]
    assert_equal(type(s2), str)
    assert_equal(type(s1), str)
    assert_equal(s1, s2)
    assert_(x1[1:1].shape == (0,))