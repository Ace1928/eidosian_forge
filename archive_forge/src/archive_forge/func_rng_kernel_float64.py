import math
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.cuda.random import \
@cuda.jit
def rng_kernel_float64(states, out, count, distribution):
    thread_id = cuda.grid(1)
    for i in range(count):
        idx = thread_id * count + i
        if distribution == UNIFORM:
            out[idx] = xoroshiro128p_uniform_float64(states, thread_id)
        elif distribution == NORMAL:
            out[idx] = xoroshiro128p_normal_float64(states, thread_id)