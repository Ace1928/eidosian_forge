import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_stream_bind(self):
    stream = cuda.stream()
    with stream.auto_synchronize():
        arr = cuda.device_array((3, 3), dtype=np.float64, stream=stream)
        self.assertEqual(arr.bind(stream).stream, stream)
        self.assertEqual(arr.stream, stream)