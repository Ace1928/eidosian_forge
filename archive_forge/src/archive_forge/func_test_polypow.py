from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polypow(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            c = np.arange(i + 1)
            tgt = reduce(poly.polymul, [c] * j, np.array([1]))
            res = poly.polypow(c, j)
            assert_equal(trim(res), trim(tgt), err_msg=msg)