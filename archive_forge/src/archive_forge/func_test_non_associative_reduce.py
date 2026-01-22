import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_non_associative_reduce(self):
    dusub = vectorize('int64(int64, int64)')(pysub)
    dudiv = vectorize('int64(int64, int64)')(pydiv)
    self._check_reduce(dusub)
    self._check_reduce_axis(dusub, dtype=np.int64)
    self._check_reduce(dudiv)
    self._check_reduce_axis(dudiv, dtype=np.int64)