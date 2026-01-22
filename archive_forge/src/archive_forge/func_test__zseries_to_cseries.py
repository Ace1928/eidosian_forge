from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test__zseries_to_cseries(self):
    for i in range(5):
        inp = np.array([0.5] * i + [2] + [0.5] * i, np.double)
        tgt = np.array([2] + [1] * i, np.double)
        res = cheb._zseries_to_cseries(inp)
        assert_equal(res, tgt)