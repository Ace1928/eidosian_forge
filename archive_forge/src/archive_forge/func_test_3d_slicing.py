import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_3d_slicing(self, flags=enable_pyobj_flags):
    pyfunc = slicing_3d_usecase
    arraytype = types.Array(types.int32, 3, 'C')
    argtys = (arraytype, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(1000, dtype='i4').reshape(10, 10, 10)
    args = [(0, 9, 1), (2, 3, 1), (9, 0, 1), (0, 9, -1), (0, 9, 2)]
    for arg in args:
        self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))
    arraytype = types.Array(types.int32, 3, 'A')
    argtys = (arraytype, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(2000, dtype='i4')[::2].reshape(10, 10, 10)
    for arg in args:
        self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))