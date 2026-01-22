import random
import numpy as np
from numba.tests.support import TestCase, captured_stdout
from numba import njit, literally
from numba.core import types
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.np.unsafe.ndarray import to_fixed_tuple, empty_inferred
from numba.core.unsafe.bytes import memcpy_region
from numba.core.unsafe.refcount import dump_refcount
from numba.cpython.unsafe.numbers import trailing_zeros, leading_zeros
from numba.core.errors import TypingError
def test_issue_3586_variant2(self):

    @njit
    def func():
        S = empty_inferred((10,))
        a = 1.1
        for i in range(S.size):
            S[i] = a + 2
        return S
    got = func()
    expect = np.asarray([3.1] * 10)
    np.testing.assert_array_equal(got, expect)