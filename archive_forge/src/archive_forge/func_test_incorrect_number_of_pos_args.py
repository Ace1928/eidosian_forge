import numpy as np
from collections import namedtuple
from numba import void, int32, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config
def test_incorrect_number_of_pos_args(self):

    @guvectorize([(int32[:], int32[:], int32[:], int32[:])], '(m),(m)->(m),(m)', target='cuda')
    def f(x, y, z, w):
        pass
    arr = np.arange(5)
    with self.assertRaises(TypeError) as te:
        f(arr)
    msg = str(te.exception)
    self.assertIn('gufunc accepts 2 positional arguments', msg)
    self.assertIn('or 4 positional arguments', msg)
    self.assertIn('Got 1 positional argument.', msg)
    with self.assertRaises(TypeError) as te:
        f(arr, arr, arr, arr, arr)
    msg = str(te.exception)
    self.assertIn('gufunc accepts 2 positional arguments', msg)
    self.assertIn('or 4 positional arguments', msg)
    self.assertIn('Got 5 positional arguments.', msg)