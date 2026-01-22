import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_setitem_whole_array_error(self):
    nbarr1 = np.recarray(1, dtype=recordwith2darray)
    nbarr2 = np.recarray(1, dtype=recordwith2darray)
    args = (nbarr1, nbarr2)
    pyfunc = record_setitem_array
    errmsg = 'Unsupported array index type'
    with self.assertRaisesRegex(TypingError, errmsg):
        self.get_cfunc(pyfunc, tuple((typeof(arg) for arg in args)))