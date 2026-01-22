from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_arrays_invalid_input(self):
    pyfunc = numpy_broadcast_arrays
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        arr = np.zeros(3, dtype=np.int64)
        s = 'hello world'
        cfunc(arr, s)
    self.assertIn('Argument "1" must be array-like', str(raises.exception))