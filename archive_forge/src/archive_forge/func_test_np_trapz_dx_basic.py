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
def test_np_trapz_dx_basic(self):
    pyfunc = np_trapz_dx
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc)
    y = [1, 2, 3]
    dx = 2
    _check({'y': y, 'dx': dx})
    y = [1, 2, 3, 4, 5]
    dx = [1, 4, 5, 6]
    _check({'y': y, 'dx': dx})
    y = [1, 2, 3, 4, 5]
    dx = [1, 4, 5, 6]
    _check({'y': y, 'dx': dx})
    y = np.linspace(-2, 5, 10)
    dx = np.nan
    _check({'y': y, 'dx': dx})
    y = np.linspace(-2, 5, 10)
    dx = np.inf
    _check({'y': y, 'dx': dx})
    y = np.linspace(-2, 5, 10)
    dx = np.linspace(-2, 5, 9)
    _check({'y': y, 'dx': dx}, abs_tol=1e-13)
    y = np.arange(60).reshape(4, 5, 3) * 1j
    dx = np.arange(40).reshape(4, 5, 2)
    _check({'y': y, 'dx': dx})
    x = np.arange(-10, 10, 0.1)
    r = cfunc(np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), dx=0.1)
    np.testing.assert_almost_equal(r, 1, 7)
    y = np.arange(20)
    dx = 1j
    _check({'y': y, 'dx': dx})
    y = np.arange(20)
    dx = np.array([5])
    _check({'y': y, 'dx': dx})