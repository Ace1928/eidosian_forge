import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def test_math_isfinite(self):
    self.unary_bool_template_float32(math_isfinite, np.isfinite)
    self.unary_bool_template_float64(math_isfinite, np.isfinite)
    self.unary_bool_template_int32(math_isfinite, np.isfinite)
    self.unary_bool_template_int64(math_isfinite, np.isfinite)
    self.unary_bool_special_values_float32(math_isfinite, np.isfinite)
    self.unary_bool_special_values_float64(math_isfinite, np.isfinite)