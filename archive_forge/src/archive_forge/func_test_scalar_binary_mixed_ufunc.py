import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def test_scalar_binary_mixed_ufunc(self):

    def _func(x, y):
        return np.add(x, y)
    vals = [2, 2, 1, 2, 0.1, 0.2]
    tys = [types.int32, types.uint32, types.int64, types.uint64, types.float32, types.float64]
    self.run_ufunc(_func, itertools.product(tys, tys), itertools.product(vals, vals))