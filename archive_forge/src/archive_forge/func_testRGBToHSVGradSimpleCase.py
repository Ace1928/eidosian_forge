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
def testRGBToHSVGradSimpleCase(self):

    def f(x):
        return gen_image_ops.rgb_to_hsv(x)
    for nptype in self.TYPES:
        x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]).astype(nptype)
        rgb_input_tensor = constant_op.constant(x, shape=x.shape)
        analytical, numerical = gradient_checker_v2.compute_gradient(f, [rgb_input_tensor])
        self.assertAllClose(numerical, analytical, atol=0.0001)