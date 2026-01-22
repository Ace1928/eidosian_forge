import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def recarray_write_array_of_nestedarray_broadcast(ary):
    ary.j[:, :, :] = 1
    return ary