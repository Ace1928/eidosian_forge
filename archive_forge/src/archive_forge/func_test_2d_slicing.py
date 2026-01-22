import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_2d_slicing(self, flags=enable_pyobj_flags):
    """
        arr_2d[a:b:c]
        """
    pyfunc = slicing_1d_usecase
    arraytype = types.Array(types.int32, 2, 'C')
    argtys = (arraytype, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(100, dtype='i4').reshape(10, 10)
    for args in [(0, 10, 1), (2, 3, 1), (10, 0, 1), (0, 10, -1), (0, 10, 2)]:
        self.assertPreciseEqual(pyfunc(a, *args), cfunc(a, *args), msg='for args %s' % (args,))