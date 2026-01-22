import numpy as np
import math
from numba import cuda, double, void
from numba.cuda.testing import unittest, CUDATestCase
@cuda.jit(void(double[:], double[:], double[:], double[:], double[:], double, double))
def black_scholes_cuda(callResult, putResult, S, X, T, R, V):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_cuda(d1)
    cndd2 = cnd_cuda(d2)
    expRT = math.exp(-1.0 * R * T[i])
    callResult[i] = S[i] * cndd1 - X[i] * expRT * cndd2
    putResult[i] = X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1)