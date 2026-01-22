import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def test_loop1_int16(self):
    pyfunc = loop1
    cfunc = njit((types.int16,))(pyfunc)
    self.assertTrue(cfunc(5), pyfunc(5))