from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polydiv(self):
    assert_raises(ZeroDivisionError, poly.polydiv, [1], [0])
    quo, rem = poly.polydiv([2], [2])
    assert_equal((quo, rem), (1, 0))
    quo, rem = poly.polydiv([2, 2], [2])
    assert_equal((quo, rem), ((1, 1), 0))
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            ci = [0] * i + [1, 2]
            cj = [0] * j + [1, 2]
            tgt = poly.polyadd(ci, cj)
            quo, rem = poly.polydiv(tgt, ci)
            res = poly.polyadd(poly.polymul(quo, ci), rem)
            assert_equal(res, tgt, err_msg=msg)