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
def test_is_real_or_complex(self):

    def values():
        yield np.array([1 + 1j, 1 + 0j, 4.5, 3, 2, 2j])
        yield np.array([1, 2, 3])
        yield 3
        yield 12j
        yield (1 + 4j)
        yield (10 + 0j)
        yield (1 + 4j, 2 + 0j)
        yield np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    pyfuncs = [iscomplex, isreal]
    for pyfunc in pyfuncs:
        cfunc = jit(nopython=True)(pyfunc)
        for x in values():
            expected = pyfunc(x)
            got = cfunc(x)
            self.assertPreciseEqual(expected, got)