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
def test_myarray_passthru(self):

    @njit
    def foo(a):
        return a
    buf = np.arange(4)
    a = MyArray(buf.shape, buf.dtype, buf)
    expected = foo.py_func(a)
    got = foo(a)
    self.assertIsInstance(got, MyArray)
    self.assertIs(type(expected), type(got))
    self.assertPreciseEqual(expected, got)