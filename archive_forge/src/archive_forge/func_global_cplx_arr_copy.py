import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def global_cplx_arr_copy(a):
    for i in range(len(a)):
        a[i] = cplx_X[i]