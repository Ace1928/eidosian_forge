from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polytrim(self):
    coef = [2, -1, 1, 0]
    assert_raises(ValueError, poly.polytrim, coef, -1)
    assert_equal(poly.polytrim(coef), coef[:-1])
    assert_equal(poly.polytrim(coef, 1), coef[:-3])
    assert_equal(poly.polytrim(coef, 2), [0])