from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermvander2d(self):
    x1, x2, x3 = self.x
    c = np.random.random((2, 3))
    van = herm.hermvander2d(x1, x2, [1, 2])
    tgt = herm.hermval2d(x1, x2, c)
    res = np.dot(van, c.flat)
    assert_almost_equal(res, tgt)
    van = herm.hermvander2d([x1], [x2], [1, 2])
    assert_(van.shape == (1, 5, 6))