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
def test_asarray_chkfinite(self):
    pyfunc = np_asarray_chkfinite
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    pairs = [(np.array([1, 2, 3]), np.float32), (np.array([1, 2, 3]),), ([1, 2, 3, 4],), (np.array([[1, 2], [3, 4]]), np.float32), (((1, 2), (3, 4)), np.int64), (np.array([1, 2], dtype=np.int64),), (np.arange(36).reshape(6, 2, 3),)]
    for pair in pairs:
        expected = pyfunc(*pair)
        got = cfunc(*pair)
        self.assertPreciseEqual(expected, got)