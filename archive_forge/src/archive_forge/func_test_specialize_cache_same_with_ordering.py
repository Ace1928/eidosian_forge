import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_specialize_cache_same_with_ordering(self):

    @cuda.jit
    def f(x, y):
        pass
    self.assertEqual(len(f.specializations), 0)
    f_f32a_f32a = f.specialize(float32[:], float32[:])
    self.assertEqual(len(f.specializations), 1)
    f_f32c_f32c = f.specialize(float32[::1], float32[::1])
    self.assertEqual(len(f.specializations), 2)
    self.assertIsNot(f_f32a_f32a, f_f32c_f32c)
    f_f32c_f32c_2 = f.specialize(float32[::1], float32[::1])
    self.assertEqual(len(f.specializations), 2)
    self.assertIs(f_f32c_f32c, f_f32c_f32c_2)