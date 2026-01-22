import numpy as np
from tensorflow.python.eager import backprop as backprop_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def testSingleClass(self):
    for label_dtype in (np.int32, np.int64):
        tf_loss, tf_gradient = self._opFwdBwd(labels=np.array([0, 0, 0]).astype(label_dtype), logits=np.array([[1.0], [-1.0], [0.0]]).astype(np.float32))
        self.assertAllClose([0.0, 0.0, 0.0], tf_loss)
        self.assertAllClose([[0.0], [0.0], [0.0]], tf_gradient)