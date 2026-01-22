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
@tf_export('image.rgb_to_yiq')
@dispatch.add_dispatch_support
def rgb_to_yiq(images):
    """Converts one or more images from RGB to YIQ.

  Outputs a tensor of the same shape as the `images` tensor, containing the YIQ
  value of the pixels.
  The output is only well defined if the value in images are in [0,1].

  Usage Example:

  >>> x = tf.constant([[[1.0, 2.0, 3.0]]])
  >>> tf.image.rgb_to_yiq(x)
  <tf.Tensor: shape=(1, 1, 3), dtype=float32,
  numpy=array([[[ 1.815     , -0.91724455,  0.09962624]]], dtype=float32)>

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
  """
    images = ops.convert_to_tensor(images, name='images')
    kernel = ops.convert_to_tensor(_rgb_to_yiq_kernel, dtype=images.dtype, name='kernel')
    ndims = images.get_shape().ndims
    return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])