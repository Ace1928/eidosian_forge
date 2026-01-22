import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
@skip_on_cudasim('Compilation unsupported in the simulator')
def test_fp16_abs_ptx(self):
    args = (f2[:], f2)
    ptx, _ = compile_ptx(simple_fp16abs, args, cc=(5, 3))
    self.assertIn('abs.f16', ptx)