from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
@skip_unless_cc_75
def test_tanhf(self):
    self._test_fast_math_unary(tanh, FastMathCriterion(fast_expected=['tanh.approx.f32 '], prec_unexpected=['tanh.approx.f32 ']))