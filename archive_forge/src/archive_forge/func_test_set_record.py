import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
@skip_on_cudasim('Structured array attr access not supported in simulator')
def test_set_record(self):
    rec = np.ones(2, dtype=recordwith2darray).view(np.recarray)[0]
    nbarr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
    arr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
    pyfunc = recarray_set_record
    pyfunc(arr, rec)
    kernel = cuda.jit(pyfunc)
    kernel[1, 1](nbarr, rec)
    np.testing.assert_equal(nbarr, arr)