import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_fill3d_threadidx(self):
    X, Y, Z = (4, 5, 6)

    def c_contigous():
        compiled = cuda.jit('void(int32[:,:,::1])')(fill3d_threadidx)
        ary = np.zeros((X, Y, Z), dtype=np.int32)
        compiled[1, (X, Y, Z)](ary)
        return ary

    def f_contigous():
        compiled = cuda.jit('void(int32[::1,:,:])')(fill3d_threadidx)
        ary = np.asfortranarray(np.zeros((X, Y, Z), dtype=np.int32))
        compiled[1, (X, Y, Z)](ary)
        return ary
    c_res = c_contigous()
    f_res = f_contigous()
    self.assertTrue(np.all(c_res == f_res))