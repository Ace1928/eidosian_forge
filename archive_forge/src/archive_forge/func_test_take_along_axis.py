import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_take_along_axis(self):
    a = np.arange(24).reshape((3, 1, 4, 2))

    @njit
    def axis_none(a, i):
        return np.take_along_axis(a, i, axis=None)
    indices = np.array([1, 2], dtype=np.uint64)
    self.assertPreciseEqual(axis_none(a, indices), axis_none.py_func(a, indices))

    def gen(axis):

        @njit
        def impl(a, i):
            return np.take_along_axis(a, i, axis)
        return impl
    for i in range(-1, a.ndim):
        jfunc = gen(i)
        ai = np.argsort(a, axis=i)
        self.assertPreciseEqual(jfunc(a, ai), jfunc.py_func(a, ai))