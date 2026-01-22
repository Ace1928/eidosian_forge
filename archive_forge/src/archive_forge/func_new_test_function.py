from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def new_test_function(self, redFunc=red_func, testArray=test_array, testName=test_name):
    ulps = 1
    if 'prod' in red_func.__name__ and np.iscomplexobj(testArray):
        ulps = 3
    npr, nbr = run_comparative(redFunc, testArray)
    self.assertPreciseEqual(npr, nbr, msg=testName, prec='single', ulps=ulps)