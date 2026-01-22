import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def test_setitem_array(self):
    N = 4
    ndim = 3
    arr = np.arange(N ** ndim).reshape((N,) * ndim).astype(np.int32) + 10
    indices = self.generate_advanced_indices(N)
    self.check_setitem_indices(arr, indices)