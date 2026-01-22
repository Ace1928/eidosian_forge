from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_array_prod_int_1d(self):
    arr = np.arange(10, dtype=np.int32) + 1
    arrty = typeof(arr)
    self.assertEqual(arrty.ndim, 1)
    self.assertEqual(arrty.layout, 'C')
    cfunc = njit((arrty,))(array_prod)
    self.assertEqual(arr.prod(), cfunc(arr))