from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def test_np_where_invalid_inputs(self):
    pyfunc = np_where_3
    cfunc = jit(nopython=True)(pyfunc)
    msg = 'The argument "condition" must be array-like'
    with self.assertRaisesRegex(TypingError, msg):
        cfunc(None, 2, 3)
    msg = 'The argument "x" must be array-like if provided'
    with self.assertRaisesRegex(TypingError, msg):
        cfunc(1, 'hello', 3)
    msg = 'The argument "y" must be array-like if provided'
    with self.assertRaisesRegex(TypingError, msg):
        cfunc(1, 2, 'world')
    msg = 'Argument "x" or "y" cannot be None'
    with self.assertRaisesRegex(TypingError, msg):
        cfunc(1, None, None)