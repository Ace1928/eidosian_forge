from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legint_zerointord(self):
    assert_equal(leg.legint((1, 2, 3), 0), (1, 2, 3))