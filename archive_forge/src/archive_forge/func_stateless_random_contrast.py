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
@tf_export('image.stateless_random_contrast', v1=[])
@dispatch.add_dispatch_support
def stateless_random_contrast(image, lower, upper, seed):
    """Adjust the contrast of images by a random factor deterministically.

  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).

  Args:
    image: An image tensor with 3 or more dimensions.
    lower: float.  Lower bound for the random contrast factor.
    upper: float.  Upper bound for the random contrast factor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)

  Usage Example:

  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...      [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_contrast(x, 0.2, 0.5, seed)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[3.4605184, 4.4605184, 5.4605184],
          [4.820173 , 5.820173 , 6.820173 ]],
         [[6.179827 , 7.179827 , 8.179828 ],
          [7.5394816, 8.539482 , 9.539482 ]]], dtype=float32)>

  Returns:
    The contrast-adjusted image(s).

  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  """
    if upper <= lower:
        raise ValueError('upper must be > lower.')
    if lower < 0:
        raise ValueError('lower must be non-negative.')
    contrast_factor = stateless_random_ops.stateless_random_uniform(shape=[], minval=lower, maxval=upper, seed=seed)
    return adjust_contrast(image, contrast_factor)