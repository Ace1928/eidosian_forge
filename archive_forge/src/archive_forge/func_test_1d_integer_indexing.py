import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_1d_integer_indexing(self, flags=enable_pyobj_flags):
    pyfunc = integer_indexing_1d_usecase
    arraytype = types.Array(types.int32, 1, 'C')
    argtys = (arraytype, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(10, dtype='i4')
    self.assertEqual(pyfunc(a, 0), cfunc(a, 0))
    self.assertEqual(pyfunc(a, 9), cfunc(a, 9))
    self.assertEqual(pyfunc(a, -1), cfunc(a, -1))
    arraytype = types.Array(types.int32, 1, 'A')
    argtys = (arraytype, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(10, dtype='i4')[::2]
    self.assertFalse(a.flags['C_CONTIGUOUS'])
    self.assertFalse(a.flags['F_CONTIGUOUS'])
    self.assertEqual(pyfunc(a, 0), cfunc(a, 0))
    self.assertEqual(pyfunc(a, 2), cfunc(a, 2))
    self.assertEqual(pyfunc(a, -1), cfunc(a, -1))
    arraytype = types.Array(types.int32, 1, 'C')
    indextype = types.Array(types.int16, 0, 'C')
    argtys = (arraytype, indextype)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(3, 13, dtype=np.int32)
    for i in (0, 9, -2):
        idx = np.array(i).astype(np.int16)
        assert idx.ndim == 0
        self.assertEqual(pyfunc(a, idx), cfunc(a, idx))