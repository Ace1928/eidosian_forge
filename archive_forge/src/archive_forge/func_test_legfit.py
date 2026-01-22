from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_legfit(self):

    def f(x):
        return x * (x - 1) * (x - 2)

    def f2(x):
        return x ** 4 + x ** 2 + 1
    assert_raises(ValueError, leg.legfit, [1], [1], -1)
    assert_raises(TypeError, leg.legfit, [[1]], [1], 0)
    assert_raises(TypeError, leg.legfit, [], [1], 0)
    assert_raises(TypeError, leg.legfit, [1], [[[1]]], 0)
    assert_raises(TypeError, leg.legfit, [1, 2], [1], 0)
    assert_raises(TypeError, leg.legfit, [1], [1, 2], 0)
    assert_raises(TypeError, leg.legfit, [1], [1], 0, w=[[1]])
    assert_raises(TypeError, leg.legfit, [1], [1], 0, w=[1, 1])
    assert_raises(ValueError, leg.legfit, [1], [1], [-1])
    assert_raises(ValueError, leg.legfit, [1], [1], [2, -1, 6])
    assert_raises(TypeError, leg.legfit, [1], [1], [])
    x = np.linspace(0, 2)
    y = f(x)
    coef3 = leg.legfit(x, y, 3)
    assert_equal(len(coef3), 4)
    assert_almost_equal(leg.legval(x, coef3), y)
    coef3 = leg.legfit(x, y, [0, 1, 2, 3])
    assert_equal(len(coef3), 4)
    assert_almost_equal(leg.legval(x, coef3), y)
    coef4 = leg.legfit(x, y, 4)
    assert_equal(len(coef4), 5)
    assert_almost_equal(leg.legval(x, coef4), y)
    coef4 = leg.legfit(x, y, [0, 1, 2, 3, 4])
    assert_equal(len(coef4), 5)
    assert_almost_equal(leg.legval(x, coef4), y)
    coef4 = leg.legfit(x, y, [2, 3, 4, 1, 0])
    assert_equal(len(coef4), 5)
    assert_almost_equal(leg.legval(x, coef4), y)
    coef2d = leg.legfit(x, np.array([y, y]).T, 3)
    assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
    coef2d = leg.legfit(x, np.array([y, y]).T, [0, 1, 2, 3])
    assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
    w = np.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = leg.legfit(x, yw, 3, w=w)
    assert_almost_equal(wcoef3, coef3)
    wcoef3 = leg.legfit(x, yw, [0, 1, 2, 3], w=w)
    assert_almost_equal(wcoef3, coef3)
    wcoef2d = leg.legfit(x, np.array([yw, yw]).T, 3, w=w)
    assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
    wcoef2d = leg.legfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
    assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
    x = [1, 1j, -1, -1j]
    assert_almost_equal(leg.legfit(x, x, 1), [0, 1])
    assert_almost_equal(leg.legfit(x, x, [0, 1]), [0, 1])
    x = np.linspace(-1, 1)
    y = f2(x)
    coef1 = leg.legfit(x, y, 4)
    assert_almost_equal(leg.legval(x, coef1), y)
    coef2 = leg.legfit(x, y, [0, 2, 4])
    assert_almost_equal(leg.legval(x, coef2), y)
    assert_almost_equal(coef1, coef2)