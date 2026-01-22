import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_broadcast_slice(self):
    nbarr = np.recarray(2, dtype=recordwith2darray)
    nbarr[0] = np.array([(1, ((1, 2), (4, 5), (2, 3)))], dtype=recordwith2darray)[0]
    nbarr[1] = np.array([(10, ((10, 20), (40, 50), (20, 30)))], dtype=recordwith2darray)[0]
    nbarr = np.broadcast_to(nbarr, (3, 2))
    funcs = (array_rec_getitem_field_slice_2d_0, array_getitem_field_slice_2d_0, array_rec_getitem_field_slice_2d_1, array_getitem_field_slice_2d_1)
    for arg, pyfunc in zip([nbarr[0], nbarr, nbarr[1], nbarr], funcs):
        ty = typeof(arg)
        arr_expected = pyfunc(arg)
        cfunc = self.get_cfunc(pyfunc, (ty,))
        arr_res = cfunc(arg)
        np.testing.assert_equal(arr_res, arr_expected)