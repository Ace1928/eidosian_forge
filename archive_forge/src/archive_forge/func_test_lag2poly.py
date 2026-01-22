from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lag2poly(self):
    for i in range(7):
        assert_almost_equal(lag.lag2poly([0] * i + [1]), Llist[i])