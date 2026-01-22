import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def record_write_full_array(rec):
    rec.j[:, :] = np.ones((3, 2))