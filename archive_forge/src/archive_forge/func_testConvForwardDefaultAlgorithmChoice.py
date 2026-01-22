import collections
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
@test_util.run_cuda_only
def testConvForwardDefaultAlgorithmChoice(self):
    in_shape = LayerShapeNCDHW(batch=2, channels=3, depth=5, height=7, width=6)
    filter_shape = FilterShape3D(depth=3, height=3, width=3, in_channels=3, out_channels=2)
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    self._assert_reproducible(lambda: nn_ops.conv3d(in_op, filter_op, strides=[1, 1, 1, 1, 1], padding='VALID', data_format='NCDHW', dilations=[1, 1, 2, 2, 2]))