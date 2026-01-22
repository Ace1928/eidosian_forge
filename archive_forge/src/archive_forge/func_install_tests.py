from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def install_tests(dtypes, funcs):
    for dt in dtypes:
        test_arrays = full_test_arrays(dt)
        for red_func, test_array in product(funcs, test_arrays):
            test_name = 'test_{0}_{1}_{2}d'
            test_name = test_name.format(red_func.__name__, test_array.dtype.name, test_array.ndim)

            def new_test_function(self, redFunc=red_func, testArray=test_array, testName=test_name):
                ulps = 1
                if 'prod' in red_func.__name__ and np.iscomplexobj(testArray):
                    ulps = 3
                npr, nbr = run_comparative(redFunc, testArray)
                self.assertPreciseEqual(npr, nbr, msg=testName, prec='single', ulps=ulps)
            setattr(cls, test_name, new_test_function)