from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legtrim(self):
    coef = [2, -1, 1, 0]
    assert_raises(ValueError, leg.legtrim, coef, -1)
    assert_equal(leg.legtrim(coef), coef[:-1])
    assert_equal(leg.legtrim(coef, 1), coef[:-3])
    assert_equal(leg.legtrim(coef, 2), [0])