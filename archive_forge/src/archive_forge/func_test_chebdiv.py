from functools import reduce
import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_chebdiv(self):
    for i in range(5):
        for j in range(5):
            msg = f'At i={i}, j={j}'
            ci = [0] * i + [1]
            cj = [0] * j + [1]
            tgt = cheb.chebadd(ci, cj)
            quo, rem = cheb.chebdiv(tgt, ci)
            res = cheb.chebadd(cheb.chebmul(quo, ci), rem)
            assert_equal(trim(res), trim(tgt), err_msg=msg)