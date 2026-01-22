import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer5(x):
    """ Test nested closure """
    y = x + 1

    def inner1(x):
        z = y + x + 2

        def inner2(x):
            return x + z
        return inner2(x) + y
    return inner1(x)