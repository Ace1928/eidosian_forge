import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
@skip_on_cudasim('Typing not done in the simulator')
def test_devicearray_typing_order_broadcasted(self):
    a = np.broadcast_to(np.array([1]), (10,))
    d = cuda.to_device(a)
    self.assertEqual(d._numba_type_.layout, 'A')