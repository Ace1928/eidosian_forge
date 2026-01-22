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
@parameterized.parameters(set((True, context.executing_eagerly())))
def testGradOnUnsupportedType(self, use_tape):
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
        in_shape = [1, 4, 6, 1]
        out_shape = [1, 2, 3, 1]
        x = np.arange(0, 24).reshape(in_shape).astype(np.uint8)
        input_tensor = constant_op.constant(x, shape=in_shape)
        tape.watch(input_tensor)
        resize_out = image_ops.resize_bicubic(input_tensor, out_shape[1:3])
        with self.cached_session():
            grad = tape.gradient(resize_out, [input_tensor])
    self.assertEqual([None], grad)