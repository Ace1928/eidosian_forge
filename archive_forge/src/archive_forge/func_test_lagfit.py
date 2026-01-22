from functools import reduce
import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
def test_lagfit(self):

    def f(x):
        return x * (x - 1) * (x - 2)
    assert_raises(ValueError, lag.lagfit, [1], [1], -1)
    assert_raises(TypeError, lag.lagfit, [[1]], [1], 0)
    assert_raises(TypeError, lag.lagfit, [], [1], 0)
    assert_raises(TypeError, lag.lagfit, [1], [[[1]]], 0)
    assert_raises(TypeError, lag.lagfit, [1, 2], [1], 0)
    assert_raises(TypeError, lag.lagfit, [1], [1, 2], 0)
    assert_raises(TypeError, lag.lagfit, [1], [1], 0, w=[[1]])
    assert_raises(TypeError, lag.lagfit, [1], [1], 0, w=[1, 1])
    assert_raises(ValueError, lag.lagfit, [1], [1], [-1])
    assert_raises(ValueError, lag.lagfit, [1], [1], [2, -1, 6])
    assert_raises(TypeError, lag.lagfit, [1], [1], [])
    x = np.linspace(0, 2)
    y = f(x)
    coef3 = lag.lagfit(x, y, 3)
    assert_equal(len(coef3), 4)
    assert_almost_equal(lag.lagval(x, coef3), y)
    coef3 = lag.lagfit(x, y, [0, 1, 2, 3])
    assert_equal(len(coef3), 4)
    assert_almost_equal(lag.lagval(x, coef3), y)
    coef4 = lag.lagfit(x, y, 4)
    assert_equal(len(coef4), 5)
    assert_almost_equal(lag.lagval(x, coef4), y)
    coef4 = lag.lagfit(x, y, [0, 1, 2, 3, 4])
    assert_equal(len(coef4), 5)
    assert_almost_equal(lag.lagval(x, coef4), y)
    coef2d = lag.lagfit(x, np.array([y, y]).T, 3)
    assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
    coef2d = lag.lagfit(x, np.array([y, y]).T, [0, 1, 2, 3])
    assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
    w = np.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = lag.lagfit(x, yw, 3, w=w)
    assert_almost_equal(wcoef3, coef3)
    wcoef3 = lag.lagfit(x, yw, [0, 1, 2, 3], w=w)
    assert_almost_equal(wcoef3, coef3)
    wcoef2d = lag.lagfit(x, np.array([yw, yw]).T, 3, w=w)
    assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
    wcoef2d = lag.lagfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
    assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
    x = [1, 1j, -1, -1j]
    assert_almost_equal(lag.lagfit(x, x, 1), [1, -1])
    assert_almost_equal(lag.lagfit(x, x, [0, 1]), [1, -1])