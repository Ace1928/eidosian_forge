import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def check_global_rec_arr(self, **jitargs):
    ctestfunc = jit(**jitargs)(global_rec_arr_copy)
    arr = np.zeros(rec_X.shape, dtype=x_dt)
    ctestfunc(arr)
    np.testing.assert_equal(arr, rec_X)