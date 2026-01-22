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
def test_float16_to_uint_ptx(self):
    pyfuncs = (to_uint8, to_uint16, to_uint32, to_uint64)
    sizes = (8, 16, 32, 64)
    for pyfunc, size in zip(pyfuncs, sizes):
        ptx, _ = compile_ptx(pyfunc, (f2,), device=True)
        self.assertIn(f'cvt.rni.u{size}.f16', ptx)