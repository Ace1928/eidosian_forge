import numpy as np
from numba import float32, jit, njit
from numba.np.ufunc import Vectorize
from numba.core.errors import TypingError
from numba.tests.support import TestCase
import unittest
def test_implicit_broadcasting(self):
    for v in vectorizers:
        vectorizer = v(add)
        vectorizer.add(float32(float32, float32))
        ufunc = vectorizer.build_ufunc()
        broadcasting_b = b[np.newaxis, :, np.newaxis, np.newaxis, :]
        self.assertPreciseEqual(ufunc(a, broadcasting_b), a + broadcasting_b)