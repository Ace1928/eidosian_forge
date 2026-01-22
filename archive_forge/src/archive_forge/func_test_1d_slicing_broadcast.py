import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_1d_slicing_broadcast(self, flags=enable_pyobj_flags):
    """
        scalar to 1d slice assignment
        """
    pyfunc = slicing_1d_usecase_set
    arraytype = types.Array(types.int32, 1, 'C')
    argtys = (arraytype, types.int16, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    N = 10
    arg = np.arange(N, dtype='i4')
    val = 42
    bounds = [0, 2, N - 2, N, N + 1, N + 3, -2, -N + 2, -N, -N - 1, -N - 3]
    for start, stop in itertools.product(bounds, bounds):
        for step in (1, 2, -1, -2):
            args = (val, start, stop, step)
            pyleft = pyfunc(arg.copy(), *args)
            cleft = cfunc(arg.copy(), *args)
            self.assertPreciseEqual(pyleft, cleft)