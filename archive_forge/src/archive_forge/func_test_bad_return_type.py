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
def test_bad_return_type(self):
    with self.assertRaises(TypeError) as te:

        @guvectorize([int32(int32[:], int32[:])], '(m)->(m)', target='cuda')
        def f(x, y):
            pass
    msg = str(te.exception)
    self.assertIn('guvectorized functions cannot return values', msg)
    self.assertIn('specifies int32 return type', msg)