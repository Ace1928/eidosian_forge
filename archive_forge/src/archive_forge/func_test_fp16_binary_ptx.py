import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
@skip_on_cudasim('Compilation unsupported in the simulator')
def test_fp16_binary_ptx(self):
    functions = (simple_fp16add, simple_fp16sub, simple_fp16mul)
    instrs = ('add.f16', 'sub.f16', 'mul.f16')
    args = (f2[:], f2, f2)
    for fn, instr in zip(functions, instrs):
        with self.subTest(instr=instr):
            ptx, _ = compile_ptx(fn, args, cc=(5, 3))
            self.assertIn(instr, ptx)