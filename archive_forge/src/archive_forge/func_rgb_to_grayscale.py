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
@tf_export('image.rgb_to_grayscale')
@dispatch.add_dispatch_support
def rgb_to_grayscale(images, name=None):
    """Converts one or more images from RGB to Grayscale.

  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 1, containing the Grayscale value of the
  pixels.

  >>> original = tf.constant([[[1.0, 2.0, 3.0]]])
  >>> converted = tf.image.rgb_to_grayscale(original)
  >>> print(converted.numpy())
  [[[1.81...]]]

  Args:
    images: The RGB tensor to convert. The last dimension must have size 3 and
      should contain RGB values.
    name: A name for the operation (optional).

  Returns:
    The converted grayscale image(s).
  """
    with ops.name_scope(name, 'rgb_to_grayscale', [images]) as name:
        images = ops.convert_to_tensor(images, name='images')
        orig_dtype = images.dtype
        flt_image = convert_image_dtype(images, dtypes.float32)
        rgb_weights = [0.2989, 0.587, 0.114]
        gray_float = math_ops.tensordot(flt_image, rgb_weights, [-1, -1])
        gray_float = array_ops.expand_dims(gray_float, -1)
        return convert_image_dtype(gray_float, orig_dtype, name=name)