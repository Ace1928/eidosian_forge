import collections
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
@test_util.run_cuda_only
def testConvTransposeForward(self, rate=1):
    in_channels = 3
    out_channels = 1
    in_shape = LayerShapeNHWC(batch=1, height=16, width=16, channels=in_channels)
    filter_shape = FilterShape2DTranspose(height=7, width=7, out_channels=out_channels, in_channels=in_channels)
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    out_shape = LayerShapeNHWC(batch=in_shape.batch, height=in_shape.height, width=in_shape.width, channels=out_channels)
    self._assert_reproducible(lambda: nn_ops.conv2d_transpose_v2(in_op, filter_op, out_shape, strides=1, padding='SAME', data_format='NHWC', dilations=[1, rate, rate, 1]))