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
def test_asarray_literal(self):

    def case1():
        return np.asarray('hello world')

    def case2():
        s = 'hello world'
        return np.asarray(s)

    def case3():
        s = '大处 着眼，小处着手。大大大处'
        return np.asarray(s)

    def case4():
        s = ''
        return np.asarray(s)
    funcs = [case1, case2, case3, case4]
    for pyfunc in funcs:
        cfunc = jit(nopython=True)(pyfunc)
        expected = pyfunc()
        got = cfunc()
        self.assertPreciseEqual(expected, got)