from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
def testCompareGpuVsCpuFloat64(self):
    in_shape = [1, 5, 7, 1]
    out_shape = [1, 9, 11, 1]
    self._gpuVsCpuCase(in_shape, out_shape, align_corners=True, half_pixel_centers=False, dtype=np.float64)