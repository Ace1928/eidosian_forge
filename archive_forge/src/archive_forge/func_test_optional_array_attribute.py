import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_optional_array_attribute(self):
    """
        Check that we can access attribute of an optional
        """

    def pyfunc(arr, do_it):
        opt = None
        if do_it:
            opt = arr
        return opt.shape[0]
    cfunc = njit(pyfunc)
    arr = np.arange(5)
    self.assertEqual(pyfunc(arr, True), cfunc(arr, True))