import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
@skip_unless_cc_53
def test_fp16_inplace_binary(self):
    functions = (simple_fp16_iadd, simple_fp16_isub, simple_fp16_imul, simple_fp16_idiv)
    ops = (operator.iadd, operator.isub, operator.imul, operator.itruediv)
    for fn, op in zip(functions, ops):
        with self.subTest(op=op):
            kernel = cuda.jit('void(f2[:], f2)')(fn)
            got = np.random.random(1).astype(np.float16)
            expected = got.copy()
            arg = np.random.random(1).astype(np.float16)[0]
            kernel[1, 1](got, arg)
            op(expected, arg)
            np.testing.assert_allclose(got, expected)