from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testInplace(self):
    y = arange(10)
    x = arange(10)
    xm = arange(10)
    xm[2] = masked
    x += 1
    assert_(eq(x, y + 1))
    xm += 1
    assert_(eq(x, y + 1))
    x = arange(10)
    xm = arange(10)
    xm[2] = masked
    x -= 1
    assert_(eq(x, y - 1))
    xm -= 1
    assert_(eq(xm, y - 1))
    x = arange(10) * 1.0
    xm = arange(10) * 1.0
    xm[2] = masked
    x *= 2.0
    assert_(eq(x, y * 2))
    xm *= 2.0
    assert_(eq(xm, y * 2))
    x = arange(10) * 2
    xm = arange(10)
    xm[2] = masked
    x //= 2
    assert_(eq(x, y))
    xm //= 2
    assert_(eq(x, y))
    x = arange(10) * 1.0
    xm = arange(10) * 1.0
    xm[2] = masked
    x /= 2.0
    assert_(eq(x, y / 2.0))
    xm /= arange(10)
    assert_(eq(xm, ones((10,))))
    x = arange(10).astype(np.float32)
    xm = arange(10)
    xm[2] = masked
    x += 1.0
    assert_(eq(x, y + 1.0))