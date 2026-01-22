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
def test_np_trapz_x_dx_basic(self):
    pyfunc = np_trapz_x_dx
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc)
    for dx in (None, 2, np.array([1, 2, 3, 4, 5])):
        y = [1, 2, 3]
        x = [4, 6, 8]
        _check({'y': y, 'x': x, 'dx': dx})
        y = [1, 2, 3, 4, 5]
        x = [4, 6]
        _check({'y': y, 'x': x, 'dx': dx})
        y = [1, 2, 3, 4, 5]
        x = [4, 5, 6, 7, 8]
        _check({'y': y, 'x': x, 'dx': dx})
        y = np.arange(60).reshape(4, 5, 3)
        self.rnd.shuffle(y)
        x = y * 1.1
        x[2, 2, 2] = np.nan
        _check({'y': y, 'x': x, 'dx': dx})