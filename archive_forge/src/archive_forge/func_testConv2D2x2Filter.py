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
def testConv2D2x2Filter(self):
    expected_output = [196, 216, 272, 296, 252, 280, 344, 376]
    self._VerifyHandValues(tensor_in_sizes=[1, 2, 3, 2], filter_in_sizes=[2, 2, 2, 2], stride=1, padding='VALID', expected=expected_output, use_gpu=False)
    self._VerifyHandValues(tensor_in_sizes=[1, 2, 3, 2], filter_in_sizes=[2, 2, 2, 2], stride=1, padding='VALID', expected=expected_output, use_gpu=True)