import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_2d_float_indexing(self, flags=enable_pyobj_flags):
    a = np.arange(100, dtype='i4').reshape(10, 10)
    pyfunc = integer_indexing_2d_usecase
    arraytype = types.Array(types.int32, 2, 'C')
    argtys = (arraytype, types.float32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    self.assertEqual(pyfunc(a, 0, 0), cfunc(a, 0, 0))
    self.assertEqual(pyfunc(a, 9, 9), cfunc(a, 9, 9))
    self.assertEqual(pyfunc(a, -1, -1), cfunc(a, -1, -1))