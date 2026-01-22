import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@skip_on_cudasim('CUDA Simulator does not force casting')
def test_explicit_signatures_device_unsafe(self):
    sigs = ['(int64, int64)']
    f = self.add_device_usecase(sigs)
    r = np.zeros(1, dtype=np.int64)
    f[1, 1](r, 1.5, 2.5)
    self.assertPreciseEqual(r[0], 3)
    self.assertEqual(len(f.overloads), 1, f.overloads)
    sigs = ['(int64, int64)', '(float64, float64)']
    f = self.add_device_usecase(sigs)
    r = np.zeros(1, dtype=np.float64)
    f[1, 1](r, np.int32(1), 2.5)
    self.assertPreciseEqual(r[0], 3.5)