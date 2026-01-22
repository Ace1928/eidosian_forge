from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyfromroots(self):
    res = poly.polyfromroots([])
    assert_almost_equal(trim(res), [1])
    for i in range(1, 5):
        roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
        tgt = Tlist[i]
        res = poly.polyfromroots(roots) * 2 ** (i - 1)
        assert_almost_equal(trim(res), trim(tgt))