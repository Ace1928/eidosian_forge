import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
@test_util.run_v1_only('b/120545219')
def testDepthwiseConv2D(self):
    for index, (input_size, filter_size, _, stride, padding, dilations) in enumerate(ConfigsToTest()):
        tf_logging.info('Testing DepthwiseConv2D, %dth config: %r * %r, stride: %d, padding: %s', index, input_size, filter_size, stride, padding)
        optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
        for data_type in [dtypes.float32] + optional_float64:
            tf_logging.info('Testing without grouped_conv')
            tolerance = 0.0001 if data_type == dtypes.float32 else 1e-12
            self._VerifyValues(input_size, filter_size, stride, padding, data_type, use_gpu=True, dilations=dilations, tolerance=tolerance)
            tf_logging.info('Testing with grouped_conv')
            self._VerifyValues(input_size, filter_size, stride, padding, data_type, use_gpu=True, grouped_conv=True, dilations=dilations, tolerance=tolerance)