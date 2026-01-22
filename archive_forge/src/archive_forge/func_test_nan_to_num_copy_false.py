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
def test_nan_to_num_copy_false(self):
    cfunc = njit(nan_to_num)
    x = np.array([0.1, 0.4, np.nan])
    expected = 1.0
    cfunc(x, copy=False, nan=expected)
    self.assertPreciseEqual(x[-1], expected)
    x_complex = np.array([0.1, 0.4, complex(np.nan, np.nan)])
    cfunc(x_complex, copy=False, nan=expected)
    self.assertPreciseEqual(x_complex[-1], 1.0 + 1j)