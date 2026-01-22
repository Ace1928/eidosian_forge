import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def test_issue9222(self):

    @njit
    def foo():

        def bar(x, y=1.1):
            return x + y
        return bar

    @njit
    def consume():
        return foo()(4)
    np.testing.assert_allclose(consume(), 4 + 1.1)