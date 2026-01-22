import unittest
from numba.tests.support import TestCase
import sys
import operator
import numpy as np
import numpy
from numba import jit, njit, typed
from numba.core import types, utils
from numba.core.errors import TypingError, LoweringError
from numba.core.types.functions import _header_lead
from numba.np.numpy_support import numpy_version
from numba.tests.support import tag, _32bit, captured_stdout
def test_array_comp_inferred_dtype_outside_setitem(self):

    def array_comp(n, v):
        arr = np.array([i for i in range(n)])
        arr[0] = v
        return arr
    v = 1.2
    self.check(array_comp, 10, v, assert_dtype=np.intp)
    with self.assertRaises(TypingError) as raises:
        cfunc = jit(nopython=True)(array_comp)
        cfunc(10, 2.3j)
    self.assertIn(_header_lead + ' Function({})'.format(operator.setitem), str(raises.exception))
    self.assertIn('(array({}, 1d, C), Literal[int](0), complex128)'.format(types.intp), str(raises.exception))