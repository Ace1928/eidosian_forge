from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_recompilation_loop(self):
    """
        https://github.com/numba/numba/issues/2481
        """
    from numba import jit

    def foo(x, y):
        A = x[::y]
        c = 1
        for k in range(A.size):
            object()
            c = c * A[::-1][k]
        return c
    cfoo = jit(forceobj=True)(foo)
    args = (np.arange(10), 1)
    self.assertEqual(foo(*args), cfoo(*args))
    self.assertEqual(len(cfoo.overloads[cfoo.signatures[0]].lifted), 1)
    lifted = cfoo.overloads[cfoo.signatures[0]].lifted[0]
    self.assertEqual(len(lifted.signatures), 1)
    args = (np.arange(10), -1)
    self.assertEqual(foo(*args), cfoo(*args))
    self.assertEqual(len(lifted.signatures), 2)