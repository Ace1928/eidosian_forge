import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_array_0d_raises(self):

    def foo(x):
        for i in x:
            pass
    with self.assertRaises(errors.TypingError) as raises:
        aryty = types.Array(types.int32, 0, 'C')
        njit((aryty,))(foo)
    self.assertIn('0-d array', str(raises.exception))