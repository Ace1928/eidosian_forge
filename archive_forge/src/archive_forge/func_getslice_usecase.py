import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
@jit(nopython=True)
def getslice_usecase(buf, i, j):
    s = buf[i:j]
    return s[0] + 2 * s[-1]