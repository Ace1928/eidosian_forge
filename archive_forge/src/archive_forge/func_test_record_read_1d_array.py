import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_record_read_1d_array(self):
    """
        Test reading from a 1D array within a structured type
        """
    rec = self.samplerec1darr.copy()
    rec['h'][0] = 4.0
    rec['h'][1] = 5.0
    nbrecord = numpy_support.from_dtype(recordwitharray)
    cfunc = self.get_cfunc(record_read_array, (nbrecord,))
    arr = np.zeros(2, dtype=rec['h'].dtype)
    cfunc[1, 1](rec, arr)
    np.testing.assert_equal(rec['h'], arr)