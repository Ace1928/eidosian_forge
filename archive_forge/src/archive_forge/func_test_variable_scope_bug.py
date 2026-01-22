from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_variable_scope_bug(self):
    """
        https://github.com/numba/numba/issues/2179

        Looplifting transformation is using the wrong version of variable `h`.
        """
    from numba import jit

    def bar(x):
        return x

    def foo(x):
        h = 0.0
        for k in range(x):
            h = h + k
        h = h - bar(x)
        return h
    cfoo = jit(forceobj=True)(foo)
    self.assertEqual(foo(10), cfoo(10))