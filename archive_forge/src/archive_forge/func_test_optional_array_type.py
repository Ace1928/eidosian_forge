import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def test_optional_array_type(self):

    @njit
    def arr_expr(x, y):
        return x + y

    @njit
    def do_call(x, y):
        if y[0] > 0:
            z = None
        else:
            z = y
        return arr_expr(x, z)
    args = (np.arange(5), np.arange(5.0))
    res = do_call(*args)
    expected = do_call.py_func(*args)
    np.testing.assert_allclose(res, expected)
    s = arr_expr.signatures
    oty = s[0][1]
    self.assertTrue(isinstance(oty, types.Optional))
    self.assertTrue(isinstance(oty.type, types.Array))
    self.assertTrue(isinstance(oty.type.dtype, types.Float))