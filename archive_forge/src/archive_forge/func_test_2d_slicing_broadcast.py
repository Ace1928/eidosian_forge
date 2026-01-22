import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_2d_slicing_broadcast(self, flags=enable_pyobj_flags):
    """
        scalar to 2d slice assignment
        """
    pyfunc = slicing_2d_usecase_set
    arraytype = types.Array(types.int32, 2, 'C')
    argtys = (arraytype, types.int16, types.int32, types.int32, types.int32, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    arg = np.arange(10 * 10, dtype='i4').reshape(10, 10)
    val = 42
    tests = [(0, 10, 1, 0, 10, 1), (2, 3, 1, 2, 3, 1), (10, 0, 1, 10, 0, 1), (0, 10, -1, 0, 10, -1), (0, 10, 2, 0, 10, 2)]
    for test in tests:
        pyleft = pyfunc(arg.copy(), val, *test)
        cleft = cfunc(arg.copy(), val, *test)
        self.assertPreciseEqual(cleft, pyleft)