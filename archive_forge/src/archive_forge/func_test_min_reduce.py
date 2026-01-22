import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_min_reduce(self):
    dumin = vectorize('int64(int64, int64)')(pymin)
    self._check_reduce(dumin, initial=10)
    self._check_reduce_axis(dumin, dtype=np.int64)