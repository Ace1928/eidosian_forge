import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
@skip_on_cudasim('numpy does not support trunc for float16')
@skip_unless_cc_53
def test_math_fp16_trunc(self):
    self.unary_template_float16(math_trunc, np.trunc)