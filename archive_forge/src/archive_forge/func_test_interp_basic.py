import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_interp_basic(self):
    pyfunc = interp
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc, abs_tol=1e-10)
    x = np.linspace(-5, 5, 25)
    xp = np.arange(-4, 8)
    fp = xp + 1.5
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    self.rnd.shuffle(x)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    self.rnd.shuffle(fp)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x[:5] = np.nan
    x[-5:] = np.inf
    self.rnd.shuffle(x)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    fp[:5] = np.nan
    fp[-5:] = -np.inf
    self.rnd.shuffle(fp)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = np.arange(-4, 8)
    xp = x + 1
    fp = x + 2
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = (2.2, 3.3, -5.0)
    xp = (2, 3, 4)
    fp = (5, 6, 7)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = ((2.2, 3.3, -5.0), (1.2, 1.3, 4.0))
    xp = np.linspace(-4, 4, 10)
    fp = np.arange(-5, 5)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = np.array([1.4, np.nan, np.inf, -np.inf, 0.0, -9.1])
    x = x.reshape(3, 2, order='F')
    xp = np.linspace(-4, 4, 10)
    fp = np.arange(-5, 5)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    for x in range(-2, 4):
        xp = [0, 1, 2]
        fp = (3, 4, 5)
        _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = np.array([])
    xp = [0, 1, 2]
    fp = (3, 4, 5)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = np.linspace(0, 25, 60).reshape(3, 4, 5)
    xp = np.arange(20)
    fp = xp - 10
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = np.nan
    xp = np.arange(5)
    fp = np.full(5, np.nan)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = np.nan
    xp = [3]
    fp = [4]
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = np.arange(-4, 8)
    xp = x
    fp = x
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = [True, False]
    xp = np.arange(-4, 8)
    fp = xp
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = [-np.inf, -1.0, 0.0, 1.0, np.inf]
    xp = np.arange(-4, 8)
    fp = xp * 2.2
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = np.linspace(-10, 10, 10)
    xp = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    fp = xp * 2.2
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = self.rnd.randn(100)
    xp = np.linspace(-3, 3, 100)
    fp = np.full(100, fill_value=3.142)
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    for factor in (1, -1):
        x = np.array([5, 6, 7]) * factor
        xp = [1, 2]
        fp = [3, 4]
        _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = 1
    xp = [1]
    fp = [True]
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    x0 = np.linspace(0, 1, 50)
    out = cfunc(x0, x, y)
    np.testing.assert_almost_equal(out, x0)
    x = np.array([1, 2, 3, 4])
    xp = np.array([1, 2, 3, 4])
    fp = np.array([1, 2, 3.01, 4])
    _check(params={'x': x, 'xp': xp, 'fp': fp})
    xp = [1]
    fp = [np.inf]
    _check(params={'x': 1, 'xp': xp, 'fp': fp})
    x = np.array([1, 2, 2.5, 3, 4])
    xp = np.array([1, 2, 3, 4])
    fp = np.array([1, 2, np.nan, 4])
    _check({'x': x, 'xp': xp, 'fp': fp})
    x = np.array([1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 5.5])
    xp = np.array([1, 2, 3, 4, 5])
    fp = np.array([np.nan, 2, np.nan, 4, np.nan])
    _check({'x': x, 'xp': xp, 'fp': fp})
    x = np.array([1, 2, 2.5, 3, 4])
    xp = np.array([1, 2, 3, 4])
    fp = np.array([1, 2, np.inf, 4])
    _check({'x': x, 'xp': xp, 'fp': fp})
    x = np.array([1, 1.5, np.nan, 2.5, -np.inf, 4, 4.5, 5, np.inf, 0, 7])
    xp = np.array([1, 2, 3, 4, 5, 6])
    fp = np.array([1, 2, np.nan, 4, 3, np.inf])
    _check({'x': x, 'xp': xp, 'fp': fp})
    x = np.array([3.10034867, 3.0999066, 3.10001529])
    xp = np.linspace(0, 10, 1 + 20000)
    fp = np.sin(xp / 2.0)
    _check({'x': x, 'xp': xp, 'fp': fp})
    x = self.rnd.uniform(0, 2 * np.pi, (100,))
    xp = np.linspace(0, 2 * np.pi, 1000)
    fp = np.cos(xp)
    exact = np.cos(x)
    got = cfunc(x, xp, fp)
    np.testing.assert_allclose(exact, got, atol=1e-05)
    x = self.rnd.randn(10)
    xp = np.linspace(-10, 10, 1000)
    fp = np.ones_like(xp)
    _check({'x': x, 'xp': xp, 'fp': fp})
    x = self.rnd.randn(1000)
    xp = np.linspace(-10, 10, 10)
    fp = np.ones_like(xp)
    _check({'x': x, 'xp': xp, 'fp': fp})