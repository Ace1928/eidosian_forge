import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_1d_slicing_add(self, flags=enable_pyobj_flags):
    pyfunc = slicing_1d_usecase_add
    arraytype = types.Array(types.int32, 1, 'C')
    argtys = (arraytype, arraytype, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    arg = np.arange(10, dtype='i4')
    for test in ((0, 10), (2, 5)):
        pyleft = pyfunc(np.zeros_like(arg), arg[slice(*test)], *test)
        cleft = cfunc(np.zeros_like(arg), arg[slice(*test)], *test)
        self.assertPreciseEqual(pyleft, cleft)