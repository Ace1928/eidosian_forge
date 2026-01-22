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
def test1x1Image(self):
    for data_format, use_gpu in [('NHWC', False), ('NCHW', False)]:
        np_input = np.arange(1.0, 129.0).reshape([4, 1, 1, 32]).astype(np.float32)
        self._testGradient(np_input, np.random.rand(32).astype(np.float32), dtypes.float32, data_format, use_gpu)