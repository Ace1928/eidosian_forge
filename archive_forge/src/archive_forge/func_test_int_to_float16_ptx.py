import numpy as np
from numba.cuda import compile_ptx
from numba.core.types import f2, i1, i2, i4, i8, u1, u2, u4, u8
from numba import cuda
from numba.core import types
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.types import float16, float32
import itertools
import unittest
@skip_on_cudasim('Compilation unsupported in the simulator')
def test_int_to_float16_ptx(self):
    fromtys = (i1, i2, i4, i8)
    sizes = (8, 16, 32, 64)
    for ty, size in zip(fromtys, sizes):
        ptx, _ = compile_ptx(to_float16, (ty,), device=True)
        self.assertIn(f'cvt.rn.f16.s{size}', ptx)