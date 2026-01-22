from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebpts2(self):
    assert_raises(ValueError, cheb.chebpts2, 1.5)
    assert_raises(ValueError, cheb.chebpts2, 1)
    tgt = [-1, 1]
    assert_almost_equal(cheb.chebpts2(2), tgt)
    tgt = [-1, 0, 1]
    assert_almost_equal(cheb.chebpts2(3), tgt)
    tgt = [-1, -0.5, 0.5, 1]
    assert_almost_equal(cheb.chebpts2(4), tgt)
    tgt = [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
    assert_almost_equal(cheb.chebpts2(5), tgt)