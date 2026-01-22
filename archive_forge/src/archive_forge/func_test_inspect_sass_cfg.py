import numpy as np
from io import StringIO
from numba import cuda, float32, float64, int32, intp
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import (skip_on_cudasim, skip_with_nvdisasm,
@skip_without_nvdisasm('nvdisasm needed for inspect_sass_cfg()')
def test_inspect_sass_cfg(self):
    sig = (float32[::1], int32[::1])

    @cuda.jit(sig)
    def add(x, y):
        i = cuda.grid(1)
        if i < len(x):
            x[i] += y[i]
    self.assertRegex(add.inspect_sass_cfg(signature=sig), 'digraph\\s*\\w\\s*{(.|\\n)*\\n}')