from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legint(self):
    assert_raises(TypeError, leg.legint, [0], 0.5)
    assert_raises(ValueError, leg.legint, [0], -1)
    assert_raises(ValueError, leg.legint, [0], 1, [0, 0])
    assert_raises(ValueError, leg.legint, [0], lbnd=[0])
    assert_raises(ValueError, leg.legint, [0], scl=[0])
    assert_raises(TypeError, leg.legint, [0], axis=0.5)
    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = leg.legint([0], m=i, k=k)
        assert_almost_equal(res, [0, 1])
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        legpol = leg.poly2leg(pol)
        legint = leg.legint(legpol, m=1, k=[i])
        res = leg.leg2poly(legint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        legpol = leg.poly2leg(pol)
        legint = leg.legint(legpol, m=1, k=[i], lbnd=-1)
        assert_almost_equal(leg.legval(-1, legint), i)
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        legpol = leg.poly2leg(pol)
        legint = leg.legint(legpol, m=1, k=[i], scl=2)
        res = leg.leg2poly(legint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = leg.legint(tgt, m=1)
            res = leg.legint(pol, m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = leg.legint(tgt, m=1, k=[k])
            res = leg.legint(pol, m=j, k=list(range(j)))
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = leg.legint(tgt, m=1, k=[k], lbnd=-1)
            res = leg.legint(pol, m=j, k=list(range(j)), lbnd=-1)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = leg.legint(tgt, m=1, k=[k], scl=2)
            res = leg.legint(pol, m=j, k=list(range(j)), scl=2)
            assert_almost_equal(trim(res), trim(tgt))