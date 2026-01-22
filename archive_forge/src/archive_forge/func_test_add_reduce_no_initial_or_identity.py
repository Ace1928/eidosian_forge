import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_add_reduce_no_initial_or_identity(self):
    duadd = vectorize('int64(int64, int64)')(pyuadd)
    self._check_reduce(duadd, dtype=np.int64)