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
def test_unary_positive_array_op_2(self):
    """
        Verify that the unary positive operator copies values, and doesn't
        just alias to the input array (mirrors normal Numpy/Python
        interaction behavior).
        """

    def f(a1):
        a2 = +a1
        a1[0] = 3
        a2[1] = 4
        return a2
    a1 = np.zeros(10)
    a2 = f(a1)
    self.assertTrue(a1[0] != a2[0] and a1[1] != a2[1])
    a3 = np.zeros(10)
    a4 = njit(f)(a3)
    self.assertTrue(a3[0] != a4[0] and a3[1] != a4[1])
    np.testing.assert_array_equal(a1, a3)
    np.testing.assert_array_equal(a2, a4)