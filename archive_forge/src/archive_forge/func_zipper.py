from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
import numpy as np
@cuda.jit
def zipper(x, y, error):
    i = 0
    for xv, yv in zip(x, y):
        if xv != x[i]:
            error[0] = 1
        if yv != y[i]:
            error[0] = 2
        i += 1
    if i != len(x):
        error[0] = 3