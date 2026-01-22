import gc
import numpy as np
import unittest
from numba import njit
from numba.core.runtime import rtsys
from numba.tests.support import TestCase, EnableNRTStatsMixin
@njit
def if_with_allocation_and_initialization(arr1, test1):
    tmp_arr = np.zeros_like(arr1)
    for i in range(tmp_arr.shape[0]):
        pass
    if test1:
        np.zeros_like(arr1)
    return tmp_arr