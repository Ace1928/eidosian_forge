import math
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.cuda.random import \
def test_create_subsequence_start(self):
    states = cuda.random.create_xoroshiro128p_states(10, seed=1)
    s1 = states.copy_to_host()
    states = cuda.random.create_xoroshiro128p_states(10, seed=1, subsequence_start=3)
    s2 = states.copy_to_host()
    np.testing.assert_array_equal(s1[3:], s2[:-3])