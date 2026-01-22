import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_is_this_a_none_objmode(self):
    pyfunc = is_this_a_none
    cfunc = jit((types.intp,), forceobj=True)(pyfunc)
    self.assertTrue(cfunc.overloads[cfunc.signatures[0]].objectmode)
    for v in [-1, 0, 1, 2]:
        self.assertPreciseEqual(pyfunc(v), cfunc(v))