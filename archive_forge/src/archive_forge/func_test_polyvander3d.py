from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyvander3d(self):
    x1, x2, x3 = self.x
    c = np.random.random((2, 3, 4))
    van = poly.polyvander3d(x1, x2, x3, [1, 2, 3])
    tgt = poly.polyval3d(x1, x2, x3, c)
    res = np.dot(van, c.flat)
    assert_almost_equal(res, tgt)
    van = poly.polyvander3d([x1], [x2], [x3], [1, 2, 3])
    assert_(van.shape == (1, 5, 24))