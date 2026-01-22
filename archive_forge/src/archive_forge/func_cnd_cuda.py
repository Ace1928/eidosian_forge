import numpy as np
import math
from numba import cuda, double, void
from numba.cuda.testing import unittest, CUDATestCase
@cuda.jit(double(double), device=True, inline=True)
def cnd_cuda(d):
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = RSQRT2PI * math.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val