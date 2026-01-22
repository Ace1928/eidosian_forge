import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_unless_cudasim
import numba.cuda.simulator as simulator
import unittest
@cuda.jit
def simple_kernel(f):
    f[1] = f[0]