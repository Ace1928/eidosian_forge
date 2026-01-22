from numba import vectorize
from numba import cuda, float32
import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@cuda.jit(float32(float32, float32, float32), device=True)
def cu_device_fn(x, y, z):
    return x ** y / z