import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
def test_cpu_dispatcher_other_module(self):

    @jit
    def add(a, b):
        return a + b
    mymod = types.ModuleType(name='mymod')
    mymod.add = add
    del add

    @cuda.jit
    def add_kernel(ary):
        i = cuda.grid(1)
        ary[i] = mymod.add(ary[i], 1)
    ary = np.arange(10)
    expect = ary + 1
    add_kernel[1, ary.size](ary)
    np.testing.assert_equal(expect, ary)