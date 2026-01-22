import ctypes
import os
import subprocess
import sys
from collections import namedtuple
import numpy as np
from numba import cfunc, carray, farray, njit
from numba.core import types, typing, utils
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import (TestCase, skip_unless_cffi, tag,
import unittest
from numba.np import numpy_support
def check_carray_farray(self, func, order):

    def eq(got, expected):
        self.assertPreciseEqual(got, expected)
        self.assertEqual(got.ctypes.data, expected.ctypes.data)
    base = np.arange(6).reshape((2, 3)).astype(np.float32).copy(order=order)
    a = func(self.make_float32_pointer(base), base.shape)
    eq(a, base)
    a = func(self.make_float32_pointer(base), base.size)
    eq(a, base.ravel('K'))
    a = func(self.make_float32_pointer(base), base.shape, base.dtype)
    eq(a, base)
    a = func(self.make_float32_pointer(base), base.shape, np.float32)
    eq(a, base)
    a = func(self.make_voidptr(base), base.shape, base.dtype)
    eq(a, base)
    a = func(self.make_voidptr(base), base.shape, np.int32)
    eq(a, base.view(np.int32))
    with self.assertRaises(TypeError):
        func(self.make_voidptr(base), base.shape)
    with self.assertRaises(TypeError):
        func(base.ctypes.data, base.shape)
    with self.assertRaises(TypeError) as raises:
        func(self.make_float32_pointer(base), base.shape, np.int32)
    self.assertIn("mismatching dtype 'int32' for pointer", str(raises.exception))