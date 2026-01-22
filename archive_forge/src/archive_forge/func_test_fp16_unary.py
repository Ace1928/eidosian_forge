import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
@skip_unless_cc_53
def test_fp16_unary(self):
    functions = (simple_fp16neg, simple_fp16abs)
    ops = (operator.neg, operator.abs)
    for fn, op in zip(functions, ops):
        with self.subTest(op=op):
            kernel = cuda.jit('void(f2[:], f2)')(fn)
            got = np.zeros(1, dtype=np.float16)
            arg1 = np.random.random(1).astype(np.float16)
            kernel[1, 1](got, arg1[0])
            expected = op(arg1)
            np.testing.assert_allclose(got, expected)