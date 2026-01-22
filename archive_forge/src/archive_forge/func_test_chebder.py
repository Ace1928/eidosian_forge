from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebder(self):
    assert_raises(TypeError, cheb.chebder, [0], 0.5)
    assert_raises(ValueError, cheb.chebder, [0], -1)
    for i in range(5):
        tgt = [0] * i + [1]
        res = cheb.chebder(tgt, m=0)
        assert_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = cheb.chebder(cheb.chebint(tgt, m=j), m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = cheb.chebder(cheb.chebint(tgt, m=j, scl=2), m=j, scl=0.5)
            assert_almost_equal(trim(res), trim(tgt))