import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_2d_integer_indexing(self, flags=enable_pyobj_flags, pyfunc=integer_indexing_2d_usecase):
    a = np.arange(100, dtype='i4').reshape(10, 10)
    arraytype = types.Array(types.int32, 2, 'C')
    argtys = (arraytype, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    self.assertEqual(pyfunc(a, 0, 3), cfunc(a, 0, 3))
    self.assertEqual(pyfunc(a, 9, 9), cfunc(a, 9, 9))
    self.assertEqual(pyfunc(a, -2, -1), cfunc(a, -2, -1))
    a = np.arange(100, dtype='i4').reshape(10, 10)[::2, ::2]
    self.assertFalse(a.flags['C_CONTIGUOUS'])
    self.assertFalse(a.flags['F_CONTIGUOUS'])
    arraytype = types.Array(types.int32, 2, 'A')
    argtys = (arraytype, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    self.assertEqual(pyfunc(a, 0, 1), cfunc(a, 0, 1))
    self.assertEqual(pyfunc(a, 2, 2), cfunc(a, 2, 2))
    self.assertEqual(pyfunc(a, -2, -1), cfunc(a, -2, -1))
    a = np.arange(100, dtype='i4').reshape(10, 10)
    arraytype = types.Array(types.int32, 2, 'C')
    indextype = types.Array(types.int32, 0, 'C')
    argtys = (arraytype, indextype, indextype)
    cfunc = jit(argtys, **flags)(pyfunc)
    for i, j in [(0, 3), (8, 9), (-2, -1)]:
        i = np.array(i).astype(np.int32)
        j = np.array(j).astype(np.int32)
        self.assertEqual(pyfunc(a, i, j), cfunc(a, i, j))