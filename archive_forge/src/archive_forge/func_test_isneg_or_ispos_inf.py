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
def test_isneg_or_ispos_inf(self):

    def values():
        yield (np.NINF, None)
        yield (np.inf, None)
        yield (np.PINF, None)
        yield (np.asarray([-np.inf, 0.0, np.inf]), None)
        yield (np.NINF, np.zeros(1, dtype=np.bool_))
        yield (np.inf, np.zeros(1, dtype=np.bool_))
        yield (np.PINF, np.zeros(1, dtype=np.bool_))
        yield (np.NINF, np.empty(12))
        yield (np.asarray([-np.inf, 0.0, np.inf]), np.zeros(3, dtype=np.bool_))
    pyfuncs = [isneginf, isposinf]
    for pyfunc in pyfuncs:
        cfunc = jit(nopython=True)(pyfunc)
        for x, out in values():
            expected = pyfunc(x, out)
            got = cfunc(x, out)
            self.assertPreciseEqual(expected, got)