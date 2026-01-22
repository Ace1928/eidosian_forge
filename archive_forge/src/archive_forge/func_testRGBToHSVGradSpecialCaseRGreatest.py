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
def testRGBToHSVGradSpecialCaseRGreatest(self):
    in_shape = [2, 10, 20, 3]

    def f(x):
        return gen_image_ops.rgb_to_hsv(x)

    def f_dummy(x):
        r = x[..., 0]
        g = x[..., 1]
        b = x[..., 2]
        v = r
        s = 1 - math_ops.div_no_nan(b, r)
        h = 60 * math_ops.div_no_nan(g - b, r - b)
        h = h / 360
        return array_ops_stack.stack([h, s, v], axis=-1)
    x_reds = np.ones((in_shape[0], in_shape[1], in_shape[2])).astype(np.float32)
    x_greens = 0.5 * np.ones((in_shape[0], in_shape[1], in_shape[2])).astype(np.float32)
    x_blues = 0.2 * np.ones((in_shape[0], in_shape[1], in_shape[2])).astype(np.float32)
    x = np.stack([x_reds, x_greens, x_blues], axis=-1)
    rgb_input_tensor = constant_op.constant(x, shape=in_shape)
    analytical, numerical = gradient_checker_v2.compute_gradient(f, [rgb_input_tensor])
    analytical_dummy, numerical_dummy = gradient_checker_v2.compute_gradient(f_dummy, [rgb_input_tensor])
    self.assertAllClose(numerical, analytical, atol=0.0001)
    self.assertAllClose(analytical_dummy, analytical, atol=0.0001)
    self.assertAllClose(numerical_dummy, numerical, atol=0.0001)