import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_unless_cudasim
import numba.cuda.simulator as simulator
import unittest
@simulator.jit
def assign_with_sync(x, y):
    i = cuda.grid(1)
    y[i] = x[i]
    cuda.syncthreads()
    cuda.syncthreads()