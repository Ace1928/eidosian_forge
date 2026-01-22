import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_clz_u4(self):
    """
        Although the CUDA Math API
        (http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html)
        only says int32 & int64 arguments are supported in C code, the LLVM
        IR input supports i8, i16, i32 & i64 (LLVM doesn't have a concept of
        unsigned integers, just unsigned operations on integers).
        http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics
        """
    compiled = cuda.jit('void(int32[:], uint32)')(simple_clz)
    ary = np.zeros(1, dtype=np.int32)
    compiled[1, 1](ary, 1048576)
    self.assertEqual(ary[0], 11)