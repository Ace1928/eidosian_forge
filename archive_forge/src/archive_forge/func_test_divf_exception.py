from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
def test_divf_exception(self):

    def f10(r, x, y):
        r[0] = x / y
    sig = (float32[::1], float32, float32)
    fastver = cuda.jit(sig, fastmath=True, debug=True)(f10)
    precver = cuda.jit(sig, debug=True)(f10)
    nelem = 10
    ary = np.empty(nelem, dtype=np.float32)
    with self.assertRaises(ZeroDivisionError):
        precver[1, nelem](ary, 10.0, 0.0)
    try:
        fastver[1, nelem](ary, 10.0, 0.0)
    except ZeroDivisionError:
        self.fail('Divide in fastmath should not throw ZeroDivisionError')