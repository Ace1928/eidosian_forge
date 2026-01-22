from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_err(a, q):
    with self.assertRaises(ValueError) as raises:
        cfunc(a, q)
    self.assertEqual('Quantiles must be in the range [0, 1]', str(raises.exception))