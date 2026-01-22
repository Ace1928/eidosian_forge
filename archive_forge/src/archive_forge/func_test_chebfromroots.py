from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebfromroots(self):
    res = cheb.chebfromroots([])
    assert_almost_equal(trim(res), [1])
    for i in range(1, 5):
        roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
        tgt = [0] * i + [1]
        res = cheb.chebfromroots(roots) * 2 ** (i - 1)
        assert_almost_equal(trim(res), trim(tgt))