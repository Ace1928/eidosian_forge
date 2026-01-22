import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
@skip_on_cudasim('Typing not done in the simulator')
def test_devicearray_typing_order_noncontig_slice_f(self):
    a = np.zeros((5, 5), order='F')
    d = cuda.to_device(a)[2, :]
    self.assertEqual(d._numba_type_.layout, 'A')