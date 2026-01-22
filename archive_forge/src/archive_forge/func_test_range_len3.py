import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def test_range_len3(self):
    pyfunc = range_len3
    typelist = [types.int16, types.int32, types.int64]
    arglist = [(1, 2, 1), (2, 8, 3), (-10, -11, -10), (-10, -10, -2)]
    for typ in typelist:
        cfunc = njit((typ, typ, typ))(pyfunc)
        for args in arglist:
            args_ = tuple((typ(x) for x in args))
            self.assertEqual(cfunc(*args_), pyfunc(*args_))