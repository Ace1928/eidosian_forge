import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def test_range_iter_list(self):
    range_iter_func = range_iter_len2
    cfunc = njit((types.List(types.intp, reflected=True),))(range_iter_func)
    arglist = [1, 2, 3, 4, 5]
    self.assertEqual(cfunc(arglist), len(arglist))