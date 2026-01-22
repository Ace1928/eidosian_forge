from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebmulx(self):
    assert_equal(cheb.chebmulx([0]), [0])
    assert_equal(cheb.chebmulx([1]), [0, 1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [0.5, 0, 0.5]
        assert_equal(cheb.chebmulx(ser), tgt)