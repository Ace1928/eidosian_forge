import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def outer16(x):
    """ closure is generator, consumed locally """
    z = x + 1

    def inner(x):
        yield (x + z)
    return list(inner(x))