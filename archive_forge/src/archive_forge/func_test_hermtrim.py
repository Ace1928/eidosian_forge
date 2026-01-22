from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermtrim(self):
    coef = [2, -1, 1, 0]
    assert_raises(ValueError, herm.hermtrim, coef, -1)
    assert_equal(herm.hermtrim(coef), coef[:-1])
    assert_equal(herm.hermtrim(coef, 1), coef[:-3])
    assert_equal(herm.hermtrim(coef, 2), [0])