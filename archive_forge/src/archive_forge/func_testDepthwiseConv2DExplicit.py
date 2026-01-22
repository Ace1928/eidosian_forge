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
def testDepthwiseConv2DExplicit(self):
    for index, (input_size, filter_size, _, stride, padding, dilations) in enumerate(ConfigsToTestExplicit()):
        tf_logging.info('Testing DepthwiseConv2D, %dth config: %r * %r, stride: %d, padding: %s', index, input_size, filter_size, stride, padding)
        data_types = [dtypes.float16, dtypes.float32]
        if not test.is_built_with_rocm():
            data_types.extend([dtypes.float64, dtypes.bfloat16])
        data_formats = ['NHWC', 'NCHW'] if test.is_gpu_available() else ['NHWC']
        for data_type in data_types:
            for data_format in data_formats:
                tolerance = 0.02 if data_type == dtypes.bfloat16 else None
                self._VerifyValues(input_size, filter_size, stride, padding, data_type, use_gpu=True, data_format=data_format, dilations=dilations, tolerance=tolerance)