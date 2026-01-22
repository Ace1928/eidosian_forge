from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebvander3d(self):
    x1, x2, x3 = self.x
    c = np.random.random((2, 3, 4))
    van = cheb.chebvander3d(x1, x2, x3, [1, 2, 3])
    tgt = cheb.chebval3d(x1, x2, x3, c)
    res = np.dot(van, c.flat)
    assert_almost_equal(res, tgt)
    van = cheb.chebvander3d([x1], [x2], [x3], [1, 2, 3])
    assert_(van.shape == (1, 5, 24))