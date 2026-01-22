import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def check_global_complex_arr(self, **jitargs):
    ctestfunc = jit(**jitargs)(global_cplx_arr_copy)
    arr = np.zeros(len(cplx_X), dtype=np.complex128)
    ctestfunc(arr)
    np.testing.assert_equal(arr, cplx_X)