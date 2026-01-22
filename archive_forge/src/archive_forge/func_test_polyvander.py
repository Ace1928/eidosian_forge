from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyvander(self):
    x = np.arange(3)
    v = poly.polyvander(x, 3)
    assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        assert_almost_equal(v[..., i], poly.polyval(x, coef))
    x = np.array([[1, 2], [3, 4], [5, 6]])
    v = poly.polyvander(x, 3)
    assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        assert_almost_equal(v[..., i], poly.polyval(x, coef))