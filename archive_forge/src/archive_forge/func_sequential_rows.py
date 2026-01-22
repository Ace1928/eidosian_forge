from __future__ import print_function
import numpy as np
from numba import config, cuda, int32
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def sequential_rows(M):
    col = cuda.grid(1)
    g = cuda.cg.this_grid()
    rows = M.shape[0]
    cols = M.shape[1]
    for row in range(1, rows):
        opposite = cols - col - 1
        M[row, col] = M[row - 1, opposite] + 1
        g.sync()