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
def testIdentityGrads(self):
    """Tests that Gradients for 1.0 scale should be ones for some kernels."""
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]
    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)
    kernel_types = ['lanczos1', 'lanczos3', 'lanczos5', 'triangle', 'keyscubic']
    scale = (1.0, 1.0)
    translation = (0.0, 0.0)
    antialias = True
    for kernel_type in kernel_types:
        with self.cached_session():
            input_tensor = constant_op.constant(x, shape=in_shape)
            with backprop.GradientTape() as tape:
                tape.watch(input_tensor)
                scale_and_translate_out = image_ops.scale_and_translate(input_tensor, out_shape[1:3], scale=constant_op.constant(scale), translation=constant_op.constant(translation), kernel_type=kernel_type, antialias=antialias)
            grad = tape.gradient(scale_and_translate_out, input_tensor)[0]
            grad_v = self.evaluate(grad)
            self.assertAllClose(np.ones_like(grad_v), grad_v)