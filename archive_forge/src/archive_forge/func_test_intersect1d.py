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
def test_intersect1d(self):

    def arrays():
        yield ([], [])
        yield ([1], [])
        yield ([], [1])
        yield ([1], [2])
        yield ([1], [1])
        yield ([1, 2], [1])
        yield ([1, 2, 2], [2, 2])
        yield ([1, 2], [2, 1])
        yield ([1, 2, 3], [1, 2, 3])
    pyfunc = intersect1d
    cfunc = jit(nopython=True)(pyfunc)
    for a, b in arrays():
        a = np.array(a)
        b = np.array(b)
        expected = pyfunc(a, b)
        got = cfunc(a, b)
        self.assertPreciseEqual(expected, got)