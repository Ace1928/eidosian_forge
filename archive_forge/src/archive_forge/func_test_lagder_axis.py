from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagder_axis(self):
    c2d = np.random.random((3, 4))
    tgt = np.vstack([lag.lagder(c) for c in c2d.T]).T
    res = lag.lagder(c2d, axis=0)
    assert_almost_equal(res, tgt)
    tgt = np.vstack([lag.lagder(c) for c in c2d])
    res = lag.lagder(c2d, axis=1)
    assert_almost_equal(res, tgt)