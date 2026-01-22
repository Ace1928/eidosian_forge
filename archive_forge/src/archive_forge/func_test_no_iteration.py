from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_no_iteration(self):
    from numba import jit

    @jit(forceobj=True)
    def test(n):
        res = 0
        for i in range(n):
            res += i
        return res
    self.assertEqual(test.py_func(-1), test(-1))
    self.assert_has_lifted(test, loopcount=1)
    self.assertEqual(test.py_func(1), test(1))
    self.assert_has_lifted(test, loopcount=1)