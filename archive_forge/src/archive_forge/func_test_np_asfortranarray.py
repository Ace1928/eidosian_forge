from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def test_np_asfortranarray(self):
    self.check_layout_dependent_func(np_asfortranarray)
    self.check_bad_array(np_asfortranarray)
    self.check_ascontiguousarray_scalar(np_asfortranarray)