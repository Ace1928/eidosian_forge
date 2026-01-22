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
@tf_export('image.image_gradients')
@dispatch.add_dispatch_support
def image_gradients(image):
    """Returns image gradients (dy, dx) for each color channel.

  Both output tensors have the same shape as the input: [batch_size, h, w,
  d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
  location (x, y). That means that dy will always have zeros in the last row,
  and dx will always have zeros in the last column.

  Usage Example:
    ```python
    BATCH_SIZE = 1
    IMAGE_HEIGHT = 5
    IMAGE_WIDTH = 5
    CHANNELS = 1
    image = tf.reshape(tf.range(IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS,
      delta=1, dtype=tf.float32),
      shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    dy, dx = tf.image.image_gradients(image)
    print(image[0, :,:,0])
    tf.Tensor(
      [[ 0.  1.  2.  3.  4.]
      [ 5.  6.  7.  8.  9.]
      [10. 11. 12. 13. 14.]
      [15. 16. 17. 18. 19.]
      [20. 21. 22. 23. 24.]], shape=(5, 5), dtype=float32)
    print(dy[0, :,:,0])
    tf.Tensor(
      [[5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [0. 0. 0. 0. 0.]], shape=(5, 5), dtype=float32)
    print(dx[0, :,:,0])
    tf.Tensor(
      [[1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]], shape=(5, 5), dtype=float32)
    ```

  Args:
    image: Tensor with shape [batch_size, h, w, d].

  Returns:
    Pair of tensors (dy, dx) holding the vertical and horizontal image
    gradients (1-step finite difference).

  Raises:
    ValueError: If `image` is not a 4D tensor.
  """
    if image.get_shape().ndims != 4:
        raise ValueError('image_gradients expects a 4D tensor [batch_size, h, w, d], not {}.'.format(image.get_shape()))
    image_shape = array_ops.shape(image)
    batch_size, height, width, depth = array_ops_stack.unstack(image_shape)
    dy = image[:, 1:, :, :] - image[:, :-1, :, :]
    dx = image[:, :, 1:, :] - image[:, :, :-1, :]
    shape = array_ops_stack.stack([batch_size, 1, width, depth])
    dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
    dy = array_ops.reshape(dy, image_shape)
    shape = array_ops_stack.stack([batch_size, height, 1, depth])
    dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
    dx = array_ops.reshape(dx, image_shape)
    return (dy, dx)