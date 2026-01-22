from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_arrays_invalid_mixed_input_types(self):
    pyfunc = numpy_broadcast_arrays
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        arr = np.arange(6).reshape((2, 3))
        b = True
        cfunc(arr, b)
    self.assertIn('Mismatch of argument types', str(raises.exception))