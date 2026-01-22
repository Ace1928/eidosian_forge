from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
import numpy as np
@cuda.jit
def enumerator_zipper(x, y, error):
    count = 0
    for i, (xv, yv) in enumerate(zip(x, y)):
        if i != count:
            error[0] = 1
        if xv != x[i]:
            error[0] = 2
        if yv != y[i]:
            error[0] = 3
        count += 1
    if count != len(x):
        error[0] = 4