from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyvandernegdeg(self):
    x = np.arange(3)
    assert_raises(ValueError, poly.polyvander, x, -1)