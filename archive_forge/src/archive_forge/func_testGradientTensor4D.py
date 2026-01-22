import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def testGradientTensor4D(self):
    for data_format, use_gpu in [('NHWC', False)]:
        for dtype in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16):
            np_input = np.arange(1.0, 49.0, dtype=dtype.as_numpy_dtype).reshape([2, 3, 4, 2]).astype(np.float32)
            bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
            self._testGradient(np_input, bias, dtype, data_format, use_gpu)
            np_input = np.arange(1.0, 513.0, dtype=dtype.as_numpy_dtype).reshape([64, 2, 2, 2]).astype(np.float32)
            self._testGradient(np_input, bias, dtype, data_format, use_gpu)
            np_input = np.arange(1.0, 513.0, dtype=dtype.as_numpy_dtype).reshape([2, 2, 2, 64]).astype(np.float32)
            self._testGradient(np_input, np.random.rand(64).astype(dtype.as_numpy_dtype), dtype, data_format, use_gpu)