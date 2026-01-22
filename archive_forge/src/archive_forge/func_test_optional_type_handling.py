import numpy as np
from numba import float32, jit, njit
from numba.np.ufunc import Vectorize
from numba.core.errors import TypingError
from numba.tests.support import TestCase
import unittest
def test_optional_type_handling(self):

    @njit
    def inner(x, y):
        if y > 2:
            z = None
        else:
            z = np.ones(4)
        return np.add(x, z)
    self.assertPreciseEqual(inner(np.arange(4), 1), np.arange(1, 5).astype(np.float64))
    with self.assertRaises(TypeError) as raises:
        inner(np.arange(4), 3)
    msg = 'expected array(float64, 1d, C), got None'
    self.assertIn(msg, str(raises.exception))