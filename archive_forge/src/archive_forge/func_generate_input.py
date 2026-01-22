import numpy as np
from numba import cuda, float32, void
from numba.cuda.testing import unittest, CUDATestCase
def generate_input(n):
    A = np.array(np.arange(n * n).reshape(n, n), dtype=np.float32)
    B = np.array(np.arange(n) + 0, dtype=A.dtype)
    return (A, B)