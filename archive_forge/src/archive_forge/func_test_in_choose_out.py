import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_in_choose_out(self):
    """Functions are passed in as arguments and returned conditionally.

        """

    def a(i):
        return i + 1

    def b(i):
        return i + 2

    def foo(a, b, choose_left):
        if choose_left:
            return a
        else:
            return b
    sig = int64(int64)
    for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
        for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
            jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(jit_(foo)(a_, b_, True).pyfunc, foo(a, b, True))
                self.assertEqual(jit_(foo)(a_, b_, False).pyfunc, foo(a, b, False))
                self.assertNotEqual(jit_(foo)(a_, b_, True).pyfunc, foo(a, b, False))