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
def test_partition_exception_a_not_array_like(self):
    pyfunc = partition
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()

    def _check(a, kth):
        with self.assertTypingError() as raises:
            cfunc(a, kth)
        self.assertIn('The first argument must be an array-like', str(raises.exception))
    _check(4, 0)
    _check('Sausages', 0)