from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagval2d(self):
    x1, x2, x3 = self.x
    y1, y2, y3 = self.y
    assert_raises(ValueError, lag.lagval2d, x1, x2[:2], self.c2d)
    tgt = y1 * y2
    res = lag.lagval2d(x1, x2, self.c2d)
    assert_almost_equal(res, tgt)
    z = np.ones((2, 3))
    res = lag.lagval2d(z, z, self.c2d)
    assert_(res.shape == (2, 3))