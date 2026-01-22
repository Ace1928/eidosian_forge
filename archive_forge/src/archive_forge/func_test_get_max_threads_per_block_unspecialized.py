import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_get_max_threads_per_block_unspecialized(self):
    N = 10

    @cuda.jit
    def simple_maxthreads(ary):
        i = cuda.grid(1)
        ary[i] = i
    arr_f32 = np.zeros(N, dtype=np.float32)
    simple_maxthreads[1, 1](arr_f32)
    sig_f32 = void(float32[::1])
    max_threads_f32 = simple_maxthreads.get_max_threads_per_block(sig_f32)
    self.assertIsInstance(max_threads_f32, int)
    self.assertGreater(max_threads_f32, 0)
    max_threads_f32_all = simple_maxthreads.get_max_threads_per_block()
    self.assertEqual(max_threads_f32_all[sig_f32.args], max_threads_f32)