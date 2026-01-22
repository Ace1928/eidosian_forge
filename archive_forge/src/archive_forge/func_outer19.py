import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer19(x):
    """ closure as arg to another closure """
    z1 = x + 1
    z2 = x + 2

    def inner(x):
        return x + z1

    def inner2(f, x):
        return f(x) + z2
    return inner2(inner, x)