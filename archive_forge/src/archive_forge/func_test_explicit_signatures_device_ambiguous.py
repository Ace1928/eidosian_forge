import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_explicit_signatures_device_ambiguous(self):
    sigs = ['(float32, float64)', '(float64, float32)', '(int64, int64)']
    f = self.add_device_usecase(sigs)
    r = np.zeros(1, dtype=np.float64)
    f[1, 1](r, 1.5, 2.5)
    self.assertPreciseEqual(r[0], 4.0)