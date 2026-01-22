import numpy as np
from numba.cuda import compile_ptx
from numba.core.types import f2, i1, i2, i4, i8, u1, u2, u4, u8
from numba import cuda
from numba.core import types
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.types import float16, float32
import itertools
import unittest
@skip_unless_cc_53
def test_float_to_uint(self):
    pyfuncs = (to_int8, to_int16, to_int32, to_int64)
    totys = (np.uint8, np.uint16, np.uint32, np.uint64)
    fromtys = (np.float16, np.float32, np.float64)
    for pyfunc, toty in zip(pyfuncs, totys):
        for fromty in fromtys:
            with self.subTest(fromty=fromty, toty=toty):
                cfunc = self._create_wrapped(pyfunc, fromty, toty)
                self.assertEqual(cfunc(12.3), pyfunc(12.3))
                self.assertEqual(cfunc(12.3), int(12.3))