import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_intrinsic_forloop_step(self):
    compiled = cuda.jit('void(int32[:,::1])')(intrinsic_forloop_step)
    ntid = (4, 3)
    nctaid = (5, 6)
    shape = (ntid[0] * nctaid[0], ntid[1] * nctaid[1])
    ary = np.empty(shape, dtype=np.int32)
    compiled[nctaid, ntid](ary)
    gridX, gridY = shape
    height, width = ary.shape
    for i, j in zip(range(ntid[0]), range(ntid[1])):
        startX, startY = (gridX + i, gridY + j)
        for x in range(startX, width, gridX):
            for y in range(startY, height, gridY):
                self.assertTrue(ary[y, x] == x + y, (ary[y, x], x + y))