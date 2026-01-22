import builtins
import unittest
from numbers import Number
from functools import wraps
import numpy as np
from llvmlite import ir
import numba
from numba import njit, typeof, objmode
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import (intrinsic, lower_builtin, overload_classmethod,
from numba.np import numpy_support
from numba.tests.support import TestCase, MemoryLeakMixin
@use_logger
def test_myarray_allocator_override(self):
    """
        Checks that our custom allocator is used
        """

    @njit
    def foo(a):
        b = a + np.arange(a.size, dtype=np.float64)
        c = a + 1j
        return (b, c)
    buf = np.arange(4, dtype=np.float64)
    a = MyArray(buf.shape, buf.dtype, buf)
    expected = foo.py_func(a)
    got = foo(a)
    self.assertPreciseEqual(got, expected)
    logged_lines = _logger
    targetctx = cpu_target.target_context
    nb_dtype = typeof(buf.dtype)
    align = targetctx.get_preferred_array_alignment(nb_dtype)
    self.assertEqual(logged_lines, [('LOG _ol_array_allocate', expected[0].nbytes, align), ('LOG _ol_array_allocate', expected[1].nbytes, align)])