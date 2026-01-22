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
def test_reciprocal_ufunc(self):
    to_skip = [types.Array(types.uint32, 1, 'C'), types.uint32, types.Array(types.int32, 1, 'C'), types.int32, types.Array(types.uint64, 1, 'C'), types.uint64, types.Array(types.int64, 1, 'C'), types.int64]
    self.basic_ufunc_test(np.reciprocal, skip_inputs=to_skip)