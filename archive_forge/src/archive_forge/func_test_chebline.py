from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebline(self):
    assert_equal(cheb.chebline(3, 4), [3, 4])