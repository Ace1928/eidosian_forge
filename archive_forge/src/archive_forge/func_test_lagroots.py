from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagroots(self):
    assert_almost_equal(lag.lagroots([1]), [])
    assert_almost_equal(lag.lagroots([0, 1]), [1])
    for i in range(2, 5):
        tgt = np.linspace(0, 3, i)
        res = lag.lagroots(lag.lagfromroots(tgt))
        assert_almost_equal(trim(res), trim(tgt))