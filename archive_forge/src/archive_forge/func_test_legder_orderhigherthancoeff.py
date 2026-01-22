from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legder_orderhigherthancoeff(self):
    c = (1, 2, 3, 4)
    assert_equal(leg.legder(c, 4), [0])