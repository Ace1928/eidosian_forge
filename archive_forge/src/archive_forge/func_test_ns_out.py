import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_ns_out(self):
    """Function is passed in via namespace scoping and returned.

        """

    def a(i):
        return i + 1

    def mkfoo(a_):

        def foo():
            return a_
        return foo
    sig = int64(int64)
    for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig), mk_ctypes_func(sig)][:-1]:
        for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
            jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(jit_(mkfoo(a_))().pyfunc, mkfoo(a)())