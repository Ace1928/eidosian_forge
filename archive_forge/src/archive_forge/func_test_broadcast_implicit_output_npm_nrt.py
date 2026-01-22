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
def test_broadcast_implicit_output_npm_nrt(self):

    def pyfunc(a0, a1):
        return np.add(a0, a1)
    input1_operands = [np.arange(3, dtype='u8'), np.arange(3 * 3, dtype='u8').reshape(3, 3), np.arange(3 * 3 * 3, dtype='u8').reshape(3, 3, 3), np.arange(3, dtype='u8').reshape(3, 1), np.arange(3, dtype='u8').reshape(1, 3), np.arange(3, dtype='u8').reshape(3, 1, 1), np.arange(3 * 3, dtype='u8').reshape(3, 3, 1), np.arange(3 * 3, dtype='u8').reshape(3, 1, 3), np.arange(3 * 3, dtype='u8').reshape(1, 3, 3)]
    input2_operands = input1_operands
    for x, y in itertools.product(input1_operands, input2_operands):
        input1_type = types.Array(types.uint64, x.ndim, 'C')
        input2_type = types.Array(types.uint64, y.ndim, 'C')
        args = (input1_type, input2_type)
        cfunc = self._compile(pyfunc, args, nrt=True)
        expected = np.add(x, y)
        result = cfunc(x, y)
        np.testing.assert_array_equal(expected, result)