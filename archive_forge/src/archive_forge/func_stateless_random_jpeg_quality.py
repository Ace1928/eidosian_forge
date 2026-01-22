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
@tf_export('image.stateless_random_jpeg_quality', v1=[])
@dispatch.add_dispatch_support
def stateless_random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed):
    """Deterministically radomize jpeg encoding quality for inducing jpeg noise.

  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).

  `min_jpeg_quality` must be in the interval `[0, 100]` and less than
  `max_jpeg_quality`.
  `max_jpeg_quality` must be in the interval `[0, 100]`.

  Usage Example:

  >>> x = tf.constant([[[1, 2, 3],
  ...                   [4, 5, 6]],
  ...                  [[7, 8, 9],
  ...                   [10, 11, 12]]], dtype=tf.uint8)
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_jpeg_quality(x, 75, 95, seed)
  <tf.Tensor: shape=(2, 2, 3), dtype=uint8, numpy=
  array([[[ 0,  4,  5],
          [ 1,  5,  6]],
         [[ 5,  9, 10],
          [ 5,  9, 10]]], dtype=uint8)>

  Args:
    image: 3D image. Size of the last dimension must be 1 or 3.
    min_jpeg_quality: Minimum jpeg encoding quality to use.
    max_jpeg_quality: Maximum jpeg encoding quality to use.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Raises:
    ValueError: if `min_jpeg_quality` or `max_jpeg_quality` is invalid.
  """
    if min_jpeg_quality < 0 or max_jpeg_quality < 0 or min_jpeg_quality > 100 or (max_jpeg_quality > 100):
        raise ValueError('jpeg encoding range must be between 0 and 100.')
    if min_jpeg_quality >= max_jpeg_quality:
        raise ValueError('`min_jpeg_quality` must be less than `max_jpeg_quality`.')
    jpeg_quality = stateless_random_ops.stateless_random_uniform(shape=[], minval=min_jpeg_quality, maxval=max_jpeg_quality, seed=seed, dtype=dtypes.int32)
    return adjust_jpeg_quality(image, jpeg_quality)