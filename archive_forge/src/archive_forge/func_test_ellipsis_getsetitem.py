import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def test_ellipsis_getsetitem(self):

    @jit(nopython=True)
    def foo(arr, v):
        arr[..., 0] = arr[..., 1]
    arr = np.arange(2)
    foo(arr, 1)
    self.assertEqual(arr[0], arr[1])