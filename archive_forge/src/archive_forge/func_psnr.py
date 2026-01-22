import functools
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('image.psnr')
@dispatch.add_dispatch_support
def psnr(a, b, max_val, name=None):
    """Returns the Peak Signal-to-Noise Ratio between a and b.

  This is intended to be used on signals (or images). Produces a PSNR value for
  each image in batch.

  The last three dimensions of input are expected to be [height, width, depth].

  Example:

  ```python
      # Read images from file.
      im1 = tf.decode_png('path/to/im1.png')
      im2 = tf.decode_png('path/to/im2.png')
      # Compute PSNR over tf.uint8 Tensors.
      psnr1 = tf.image.psnr(im1, im2, max_val=255)

      # Compute PSNR over tf.float32 Tensors.
      im1 = tf.image.convert_image_dtype(im1, tf.float32)
      im2 = tf.image.convert_image_dtype(im2, tf.float32)
      psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
      # psnr1 and psnr2 both have type tf.float32 and are almost equal.
  ```

  Args:
    a: First set of images.
    b: Second set of images.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    name: Namespace to embed the computation in.

  Returns:
    The scalar PSNR between a and b. The returned tensor has type `tf.float32`
    and shape [batch_size, 1].
  """
    with ops.name_scope(name, 'PSNR', [a, b]):
        max_val = math_ops.cast(max_val, a.dtype)
        max_val = convert_image_dtype(max_val, dtypes.float32)
        a = convert_image_dtype(a, dtypes.float32)
        b = convert_image_dtype(b, dtypes.float32)
        mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-3, -2, -1])
        psnr_val = math_ops.subtract(20 * math_ops.log(max_val) / math_ops.log(10.0), np.float32(10 / np.log(10)) * math_ops.log(mse), name='psnr')
        _, _, checks = _verify_compatible_image_shapes(a, b)
        with ops.control_dependencies(checks):
            return array_ops.identity(psnr_val)