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
def test_nan_to_num(self):
    values = [np.nan, 1, 1.1, 1 + 1j, complex(-np.inf, np.nan), complex(np.nan, np.nan), np.array([1], dtype=int), np.array([complex(-np.inf, np.inf), complex(1, np.nan), complex(np.nan, 1), complex(np.inf, -np.inf)]), np.array([0.1, 1.0, 0.4]), np.array([1, 2, 3]), np.array([[0.1, 1.0, 0.4], [0.4, 1.2, 4.0]]), np.array([0.1, np.nan, 0.4]), np.array([[0.1, np.nan, 0.4], [np.nan, 1.2, 4.0]]), np.array([-np.inf, np.nan, np.inf]), np.array([-np.inf, np.nan, np.inf], dtype=np.float32)]
    nans = [0.0, 10]
    pyfunc = nan_to_num
    cfunc = njit(nan_to_num)
    for value, nan in product(values, nans):
        expected = pyfunc(value, nan=nan)
        got = cfunc(value, nan=nan)
        self.assertPreciseEqual(expected, got)