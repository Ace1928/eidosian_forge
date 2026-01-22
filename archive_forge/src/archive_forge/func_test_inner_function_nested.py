import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def test_inner_function_nested(self):

    def outer(x):

        def inner(y):

            def innermost(z):
                return x + y + z
            s = 0
            for i in range(y):
                s += innermost(i)
            return s
        return inner(x * x)
    cfunc = njit(outer)
    self.assertEqual(cfunc(10), outer(10))