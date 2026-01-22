from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_conditionally_defined_in_loop(self):
    from numba import jit

    @jit(forceobj=True)
    def test():
        x = 5
        y = 0
        for i in range(2):
            if i > 0:
                x = 6
            y += x
        return (y, x)
    self.assertEqual(test.py_func(), test())
    self.assert_has_lifted(test, loopcount=1)