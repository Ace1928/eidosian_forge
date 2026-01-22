from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebroots(self):
    assert_almost_equal(cheb.chebroots([1]), [])
    assert_almost_equal(cheb.chebroots([1, 2]), [-0.5])
    for i in range(2, 5):
        tgt = np.linspace(-1, 1, i)
        res = cheb.chebroots(cheb.chebfromroots(tgt))
        assert_almost_equal(trim(res), trim(tgt))