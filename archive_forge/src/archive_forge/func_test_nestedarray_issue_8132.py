import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_nestedarray_issue_8132(self):
    data = np.arange(27 * 2, dtype=np.float64).reshape(27, 2)
    recty = types.Record.make_c_struct([('data', types.NestedArray(dtype=types.float64, shape=data.shape))])
    arr = np.array((data,), dtype=recty.dtype)
    [extracted_array] = arr.tolist()
    np.testing.assert_array_equal(extracted_array, data)