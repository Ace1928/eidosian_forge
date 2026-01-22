from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
def test_sinf(self):
    self._test_fast_math_unary(sin, FastMathCriterion(fast_expected=['sin.approx.ftz.f32 '], prec_unexpected=['sin.approx.ftz.f32 ']))