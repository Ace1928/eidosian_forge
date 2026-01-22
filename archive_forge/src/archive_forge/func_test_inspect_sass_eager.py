import numpy as np
from io import StringIO
from numba import cuda, float32, float64, int32, intp
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import (skip_on_cudasim, skip_with_nvdisasm,
@skip_without_nvdisasm('nvdisasm needed for inspect_sass()')
def test_inspect_sass_eager(self):
    sig = (float32[::1], int32[::1])

    @cuda.jit(sig, lineinfo=True)
    def add(x, y):
        i = cuda.grid(1)
        if i < len(x):
            x[i] += y[i]
    self._test_inspect_sass(add, 'add', add.inspect_sass(sig))