import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def test_inner_function_with_closure_2(self):

    def outer(x):
        y = x + 1

        def inner(x):
            return x * y
        y = inner(x)
        return y + inner(x)
    cfunc = njit(outer)
    self.assertEqual(cfunc(10), outer(10))