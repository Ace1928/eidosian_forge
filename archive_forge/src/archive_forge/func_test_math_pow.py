import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def test_math_pow(self):
    self.binary_template_float32(math_pow, np.power)
    self.binary_template_float64(math_pow, np.power)
    self.pow_template_int32(np.float32)
    self.pow_template_int32(np.float64)