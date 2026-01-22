import numpy as np
from numba import from_dtype, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@skip_on_cudasim('Simulator does not check alignment')
def test_record_alignment_error(self):
    rec_dtype = np.dtype([('a', 'int32'), ('b', 'float64')])
    rec = from_dtype(rec_dtype)
    with self.assertRaises(Exception) as raises:

        @cuda.jit((rec[:],))
        def foo(a):
            i = cuda.grid(1)
            a[i].a = a[i].b
    self.assertTrue('type float64 is not aligned' in str(raises.exception))