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
def test_diagflat_basic(self):
    pyfunc1 = diagflat1
    cfunc1 = njit(pyfunc1)
    pyfunc2 = diagflat2
    cfunc2 = njit(pyfunc2)

    def inputs():
        yield (np.array([1, 2]), 1)
        yield (np.array([[1, 2], [3, 4]]), -2)
        yield (np.arange(8).reshape((2, 2, 2)), 2)
        yield ([1, 2], 1)
        yield (np.array([]), 1)
    for v, k in inputs():
        self.assertPreciseEqual(pyfunc1(v), cfunc1(v))
        self.assertPreciseEqual(pyfunc2(v, k), cfunc2(v, k))