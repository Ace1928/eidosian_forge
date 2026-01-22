from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_hermint(self):
    assert_raises(TypeError, herm.hermint, [0], 0.5)
    assert_raises(ValueError, herm.hermint, [0], -1)
    assert_raises(ValueError, herm.hermint, [0], 1, [0, 0])
    assert_raises(ValueError, herm.hermint, [0], lbnd=[0])
    assert_raises(ValueError, herm.hermint, [0], scl=[0])
    assert_raises(TypeError, herm.hermint, [0], axis=0.5)
    for i in range(2, 5):
        k = [0] * (i - 2) + [1]
        res = herm.hermint([0], m=i, k=k)
        assert_almost_equal(res, [0, 0.5])
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [1 / scl]
        hermpol = herm.poly2herm(pol)
        hermint = herm.hermint(hermpol, m=1, k=[i])
        res = herm.herm2poly(hermint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        hermpol = herm.poly2herm(pol)
        hermint = herm.hermint(hermpol, m=1, k=[i], lbnd=-1)
        assert_almost_equal(herm.hermval(-1, hermint), i)
    for i in range(5):
        scl = i + 1
        pol = [0] * i + [1]
        tgt = [i] + [0] * i + [2 / scl]
        hermpol = herm.poly2herm(pol)
        hermint = herm.hermint(hermpol, m=1, k=[i], scl=2)
        res = herm.herm2poly(hermint)
        assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = herm.hermint(tgt, m=1)
            res = herm.hermint(pol, m=j)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = herm.hermint(tgt, m=1, k=[k])
            res = herm.hermint(pol, m=j, k=list(range(j)))
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = herm.hermint(tgt, m=1, k=[k], lbnd=-1)
            res = herm.hermint(pol, m=j, k=list(range(j)), lbnd=-1)
            assert_almost_equal(trim(res), trim(tgt))
    for i in range(5):
        for j in range(2, 5):
            pol = [0] * i + [1]
            tgt = pol[:]
            for k in range(j):
                tgt = herm.hermint(tgt, m=1, k=[k], scl=2)
            res = herm.hermint(pol, m=j, k=list(range(j)), scl=2)
            assert_almost_equal(trim(res), trim(tgt))