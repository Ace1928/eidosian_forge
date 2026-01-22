import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
def check_local_func_lifetime(self, **jitargs):

    def f(x):
        return x + 1
    c_f = jit('int32(int32)', **jitargs)(f)
    self.assertPreciseEqual(c_f(1), 2)
    cfunc = self.get_impl(c_f)
    refs = [weakref.ref(obj) for obj in (f, c_f, cfunc.__self__)]
    obj = f = c_f = cfunc = None
    gc.collect()
    self.assertEqual([wr() for wr in refs], [None] * len(refs))