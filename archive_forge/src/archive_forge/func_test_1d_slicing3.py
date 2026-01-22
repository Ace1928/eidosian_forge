import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_1d_slicing3(self, flags=enable_pyobj_flags):
    pyfunc = slicing_1d_usecase3
    arraytype = types.Array(types.int32, 1, 'C')
    argtys = (arraytype, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(10, dtype='i4')
    args = [(3, 10), (2, 3), (10, 0), (0, 10), (5, 10)]
    for arg in args:
        self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))
    arraytype = types.Array(types.int32, 1, 'A')
    argtys = (arraytype, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(20, dtype='i4')[::2]
    self.assertFalse(a.flags['C_CONTIGUOUS'])
    self.assertFalse(a.flags['F_CONTIGUOUS'])
    for arg in args:
        self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))