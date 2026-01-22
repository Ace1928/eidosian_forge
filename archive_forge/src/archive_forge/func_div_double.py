import numpy as np
from numba import cuda, float32, float64, int32, void
from numba.cuda.testing import unittest, CUDATestCase
@cuda.jit(void(float64[:, :], int32, int32))
def div_double(grid, l_x, l_y):
    for x in range(l_x):
        for y in range(l_y):
            grid[x, y] /= 2.0