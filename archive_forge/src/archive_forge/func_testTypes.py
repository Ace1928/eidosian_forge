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
def testTypes(self):
    in_shape = [1, 4, 6, 1]
    out_shape = [1, 2, 3, 1]
    for use_gpu in [False, True]:
        for dtype in [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]:
            jacob_a, jacob_n = self._getJacobians(in_shape, out_shape, dtype=dtype, use_gpu=use_gpu)
            if dtype in (np.float16, dtypes.bfloat16.as_numpy_dtype):
                _, jacob_n = self._getJacobians(in_shape, out_shape, dtype=np.float32, use_gpu=use_gpu)
            threshold = 0.001
            if dtype == np.float64:
                threshold = 1e-05
            self.assertAllClose(jacob_a, jacob_n, threshold, threshold)