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
@tf_export('image.grayscale_to_rgb')
@dispatch.add_dispatch_support
def grayscale_to_rgb(images, name=None):
    """Converts one or more images from Grayscale to RGB.

  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 3, containing the RGB value of the pixels.
  The input images' last dimension must be size 1.

  >>> original = tf.constant([[[1.0], [2.0], [3.0]]])
  >>> converted = tf.image.grayscale_to_rgb(original)
  >>> print(converted.numpy())
  [[[1. 1. 1.]
    [2. 2. 2.]
    [3. 3. 3.]]]

  Args:
    images: The Grayscale tensor to convert. The last dimension must be size 1.
    name: A name for the operation (optional).

  Returns:
    The converted grayscale image(s).
  """
    with ops.name_scope(name, 'grayscale_to_rgb', [images]) as name:
        images = _AssertGrayscaleImage(images)
        images = ops.convert_to_tensor(images, name='images')
        rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0)
        shape_list = [array_ops.ones(rank_1, dtype=dtypes.int32)] + [array_ops.expand_dims(3, 0)]
        multiples = array_ops.concat(shape_list, 0)
        rgb = array_ops.tile(images, multiples, name=name)
        rgb.set_shape(images.get_shape()[:-1].concatenate([3]))
        return rgb