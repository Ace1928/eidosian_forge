import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer6(x):
    """ Test closure with list comprehension in body """
    y = x + 1

    def inner1(x):
        z = y + x + 2
        return [t for t in range(z)]
    return inner1(x)