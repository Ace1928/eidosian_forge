import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_unless_cc_53
def test_fp16_comparison(self):
    fns = (simple_heq_scalar, simple_hne_scalar, simple_hge_scalar, simple_hgt_scalar, simple_hle_scalar, simple_hlt_scalar)
    ops = (operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt)
    for fn, op in zip(fns, ops):
        with self.subTest(op=op):
            kernel = cuda.jit('void(b1[:], f2, f2)')(fn)
            expected = np.zeros(1, dtype=np.bool8)
            got = np.zeros(1, dtype=np.bool8)
            arg2 = np.float16(2)
            arg3 = np.float16(3)
            arg4 = np.float16(4)
            kernel[1, 1](got, arg3, arg3)
            expected = op(arg3, arg3)
            self.assertEqual(expected, got[0])
            kernel[1, 1](got, arg3, arg4)
            expected = op(arg3, arg4)
            self.assertEqual(expected, got[0])
            kernel[1, 1](got, arg3, arg2)
            expected = op(arg3, arg2)
            self.assertEqual(expected, got[0])