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
def test_gufunc_name(self):
    gufunc = _get_matmulcore_gufunc()
    self.assertEqual(gufunc.__name__, 'matmulcore')