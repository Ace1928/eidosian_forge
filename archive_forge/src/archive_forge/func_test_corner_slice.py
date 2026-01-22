import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_corner_slice(self):
    nbarr = np.recarray((1, 2, 3, 5, 7, 13, 17), dtype=recordwith4darray, order='F')
    np.random.seed(1)
    for index, _ in np.ndenumerate(nbarr):
        nbarr[index].p = np.random.randint(0, 1000, (3, 2, 5, 7), np.int64).astype(np.float32)
    funcs = (rec_getitem_range_slice_4d, recarray_getitem_range_slice_4d)
    for arg, pyfunc in zip([nbarr[0], nbarr], funcs):
        ty = typeof(arg)
        arr_expected = pyfunc(arg)
        cfunc = self.get_cfunc(pyfunc, (ty,))
        arr_res = cfunc(arg)
        np.testing.assert_equal(arr_res, arr_expected)