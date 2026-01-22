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
def test_digitize_non_monotonic_bins(self):
    self.disable_leak_check()
    pyfunc = digitize
    cfunc = jit(nopython=True)(pyfunc)

    def check_error(*args):
        for fn in (pyfunc, cfunc):
            with self.assertRaises(ValueError) as raises:
                fn(*args)
            msg = 'bins must be monotonically increasing or decreasing'
            self.assertIn(msg, str(raises.exception))
    x = np.array([np.nan, 1])
    bins = np.array([np.nan, 1.5, 2.3, np.nan])
    check_error(x, bins)
    x = [-1, 0, 1, 2]
    bins = [0, 0, 1, 0]
    check_error(x, bins)
    bins = [1, 1, 0, 1]
    check_error(x, bins)