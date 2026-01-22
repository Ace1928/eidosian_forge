from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebder_axis(self):
    c2d = np.random.random((3, 4))
    tgt = np.vstack([cheb.chebder(c) for c in c2d.T]).T
    res = cheb.chebder(c2d, axis=0)
    assert_almost_equal(res, tgt)
    tgt = np.vstack([cheb.chebder(c) for c in c2d])
    res = cheb.chebder(c2d, axis=1)
    assert_almost_equal(res, tgt)