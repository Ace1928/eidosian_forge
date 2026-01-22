import math
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.cuda.random import \
def test_create_stream(self):
    stream = cuda.stream()
    states = cuda.random.create_xoroshiro128p_states(10, seed=1, stream=stream)
    s = states.copy_to_host()
    self.assertEqual(len(np.unique(s)), 10)