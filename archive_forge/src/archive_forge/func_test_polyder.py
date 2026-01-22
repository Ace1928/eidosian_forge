from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyder(self):
    assert_raises(TypeError, poly.polyder, [0], 0.5)
    assert_raises(ValueError, poly.polyder, [0], -1)
    for i in range(5):
        tgt = [0] * i + [1]
        res = poly.polyder(tgt, m=0)
        assert_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = poly.polyder(poly.polyint(tgt, m=j), m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = poly.polyder(poly.polyint(tgt, m=j, scl=2), m=j, scl=0.5)
            assert_almost_equal(trim(res), trim(tgt))