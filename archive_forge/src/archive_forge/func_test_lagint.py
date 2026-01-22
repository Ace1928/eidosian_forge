from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagint(self):
    assert_raises(TypeError, lag.lagint, [0], 0.5)
    assert_raises(ValueError, lag.lagint, [0], -1)
    assert_raises(ValueError, lag.lagint, [0], 1, [0, 0])
    assert_raises(ValueError, lag.lagint, [0], lbnd=[0])
    assert_raises(ValueError, lag.lagint, [0], scl=[0])
    assert_raises(TypeError, lag.lagint, [0], axis=0.5)
    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = lag.lagint([0], m=i, k=k)
        assert_almost_equal(res, [1, -1])
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        lagpol = lag.poly2lag(pol)
        lagint = lag.lagint(lagpol, m=1, k=[i])
        res = lag.lag2poly(lagint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        lagpol = lag.poly2lag(pol)
        lagint = lag.lagint(lagpol, m=1, k=[i], lbnd=-1)
        assert_almost_equal(lag.lagval(-1, lagint), i)
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        lagpol = lag.poly2lag(pol)
        lagint = lag.lagint(lagpol, m=1, k=[i], scl=2)
        res = lag.lag2poly(lagint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = lag.lagint(tgt, m=1)
            res = lag.lagint(pol, m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = lag.lagint(tgt, m=1, k=[k])
            res = lag.lagint(pol, m=j, k=list(range(j)))
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = lag.lagint(tgt, m=1, k=[k], lbnd=-1)
            res = lag.lagint(pol, m=j, k=list(range(j)), lbnd=-1)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = lag.lagint(tgt, m=1, k=[k], scl=2)
            res = lag.lagint(pol, m=j, k=list(range(j)), scl=2)
            assert_almost_equal(trim(res), trim(tgt))