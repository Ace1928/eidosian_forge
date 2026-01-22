from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_issue_734(self):
    from numba import jit, void, int32, double

    @jit(void(int32, double[:]), forceobj=True)
    def forloop_with_if(u, a):
        if u == 0:
            for i in range(a.shape[0]):
                a[i] = a[i] * 2.0
        else:
            for i in range(a.shape[0]):
                a[i] = a[i] + 1.0
    for u in (0, 1):
        nb_a = np.arange(10, dtype='int32')
        np_a = np.arange(10, dtype='int32')
        forloop_with_if(u, nb_a)
        forloop_with_if.py_func(u, np_a)
        self.assertPreciseEqual(nb_a, np_a)