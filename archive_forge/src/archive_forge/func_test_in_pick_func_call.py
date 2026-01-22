import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_in_pick_func_call(self):
    """Functions are passed in as items of tuple argument, retrieved via
        indexing, and called.

        """

    def a(i):
        return i + 1

    def b(i):
        return i + 2

    def foo(funcs, i):
        f = funcs[i]
        r = f(123)
        return r
    sig = int64(int64)
    for decor in [mk_cfunc_func(sig), mk_wap_func(sig), mk_njit_with_sig_func(sig)]:
        for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
            jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(jit_(foo)((a_, b_), 0), foo((a, b), 0))
                self.assertEqual(jit_(foo)((a_, b_), 1), foo((a, b), 1))
                self.assertNotEqual(jit_(foo)((a_, b_), 0), foo((a, b), 1))