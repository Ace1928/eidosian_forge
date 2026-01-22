import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
def test_record_real(self):
    rectyp = np.dtype([('real', np.float32), ('imag', np.complex64)])
    arr = np.zeros(3, dtype=rectyp)
    arr['real'] = np.random.random(arr.size)
    arr['imag'] = np.random.random(arr.size) * 1.3j
    self.assertIs(array_real(arr), arr)
    self.assertEqual(array_imag(arr).tolist(), np.zeros_like(arr).tolist())
    jit_array_real = njit(array_real)
    jit_array_imag = njit(array_imag)
    with self.assertRaises(TypingError) as raises:
        jit_array_real(arr)
    self.assertIn('cannot access .real of array of Record', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        jit_array_imag(arr)
    self.assertIn('cannot access .imag of array of Record', str(raises.exception))