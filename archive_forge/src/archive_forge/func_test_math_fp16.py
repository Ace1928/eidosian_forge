import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
@skip_unless_cc_53
def test_math_fp16(self):
    self.unary_template_float16(math_sin, np.sin)
    self.unary_template_float16(math_cos, np.cos)
    self.unary_template_float16(math_exp, np.exp)
    self.unary_template_float16(math_log, np.log, start=1)
    self.unary_template_float16(math_log2, np.log2, start=1)
    self.unary_template_float16(math_log10, np.log10, start=1)
    self.unary_template_float16(math_fabs, np.fabs, start=-1)
    self.unary_template_float16(math_sqrt, np.sqrt)
    self.unary_template_float16(math_ceil, np.ceil)
    self.unary_template_float16(math_floor, np.floor)