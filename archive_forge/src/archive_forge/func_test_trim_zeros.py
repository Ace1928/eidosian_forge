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
def test_trim_zeros(self):

    def arrays():
        yield np.array([])
        yield np.zeros(5)
        yield np.zeros(1)
        yield np.array([1, 2, 3])
        yield np.array([0, 1, 2, 3])
        yield np.array([0.0, 1.0, 2.0, np.nan, 0.0])
        yield np.array(['0', 'Hello', 'world'])

    def explicit_trim():
        yield (np.array([0, 1, 2, 0, 0]), 'FB')
        yield (np.array([0, 1, 2]), 'B')
        yield (np.array([np.nan, 0.0, 1.2, 2.3, 0.0]), 'b')
        yield (np.array([0, 0, 1, 2, 5]), 'f')
        yield (np.array([0, 1, 2, 0]), 'abf')
        yield (np.array([0, 4, 0]), 'd')
        yield (np.array(['\x00', '1', '2']), 'f')
    pyfunc = np_trim_zeros
    cfunc = jit(nopython=True)(pyfunc)
    for arr in arrays():
        expected = pyfunc(arr)
        got = cfunc(arr)
        self.assertPreciseEqual(expected, got)
    for arr, trim in explicit_trim():
        expected = pyfunc(arr, trim)
        got = cfunc(arr, trim)
        self.assertPreciseEqual(expected, got)