import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_1d_slicing4(self, flags=enable_pyobj_flags):
    pyfunc = slicing_1d_usecase4
    arraytype = types.Array(types.int32, 1, 'C')
    argtys = (arraytype,)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(10, dtype='i4')
    self.assertEqual(pyfunc(a), cfunc(a))
    arraytype = types.Array(types.int32, 1, 'A')
    argtys = (arraytype,)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(20, dtype='i4')[::2]
    self.assertFalse(a.flags['C_CONTIGUOUS'])
    self.assertFalse(a.flags['F_CONTIGUOUS'])
    self.assertEqual(pyfunc(a), cfunc(a))