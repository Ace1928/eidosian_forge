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
@tf_export('image.stateless_random_hue', v1=[])
@dispatch.add_dispatch_support
def stateless_random_hue(image, max_delta, seed):
    """Adjust the hue of RGB images by a random factor deterministically.

  Equivalent to `adjust_hue()` but uses a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.

  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).

  `max_delta` must be in the interval `[0, 0.5]`.

  Usage Example:

  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...      [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_hue(x, 0.2, seed)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[ 1.6514902,  1.       ,  3.       ],
          [ 4.65149  ,  4.       ,  6.       ]],
         [[ 7.65149  ,  7.       ,  9.       ],
          [10.65149  , 10.       , 12.       ]]], dtype=float32)>

  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    max_delta: float. The maximum value for the random delta.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`.

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Raises:
    ValueError: if `max_delta` is invalid.
  """
    if max_delta > 0.5:
        raise ValueError('max_delta must be <= 0.5.')
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')
    delta = stateless_random_ops.stateless_random_uniform(shape=[], minval=-max_delta, maxval=max_delta, seed=seed)
    return adjust_hue(image, delta)