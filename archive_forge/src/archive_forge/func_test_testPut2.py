from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testPut2(self):
    d = arange(5)
    x = array(d, mask=[0, 0, 0, 0, 0])
    z = array([10, 40], mask=[1, 0])
    assert_(x[2] is not masked)
    assert_(x[3] is not masked)
    x[2:4] = z
    assert_(x[2] is masked)
    assert_(x[3] is not masked)
    assert_(eq(x, [0, 1, 10, 40, 4]))
    d = arange(5)
    x = array(d, mask=[0, 0, 0, 0, 0])
    y = x[2:4]
    z = array([10, 40], mask=[1, 0])
    assert_(x[2] is not masked)
    assert_(x[3] is not masked)
    y[:] = z
    assert_(y[0] is masked)
    assert_(y[1] is not masked)
    assert_(eq(y, [10, 40]))
    assert_(x[2] is masked)
    assert_(x[3] is not masked)
    assert_(eq(x, [0, 1, 10, 40, 4]))