import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_in__(self):
    """Function is passed in as an argument.
        """

    def a(i):
        return i + 1

    def foo(f):
        return 0
    sig = int64(int64)
    for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_ctypes_func(sig), mk_wap_func(sig)]:
        for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
            jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__, jit=jit_opts):
                a_ = decor(a)
                self.assertEqual(jit_(foo)(a_), foo(a))