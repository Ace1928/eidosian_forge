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
@tf_export('image.sobel_edges')
@dispatch.add_dispatch_support
def sobel_edges(image):
    """Returns a tensor holding Sobel edge maps.

  Example usage:

  For general usage, `image` would be loaded from a file as below:

  ```python
  image_bytes = tf.io.read_file(path_to_image_file)
  image = tf.image.decode_image(image_bytes)
  image = tf.cast(image, tf.float32)
  image = tf.expand_dims(image, 0)
  ```
  But for demo purposes, we are using randomly generated values for `image`:

  >>> image = tf.random.uniform(
  ...   maxval=255, shape=[1, 28, 28, 3], dtype=tf.float32)
  >>> sobel = tf.image.sobel_edges(image)
  >>> sobel_y = np.asarray(sobel[0, :, :, :, 0]) # sobel in y-direction
  >>> sobel_x = np.asarray(sobel[0, :, :, :, 1]) # sobel in x-direction

  For displaying the sobel results, PIL's [Image Module](
  https://pillow.readthedocs.io/en/stable/reference/Image.html) can be used:

  ```python
  # Display edge maps for the first channel (at index 0)
  Image.fromarray(sobel_y[..., 0] / 4 + 0.5).show()
  Image.fromarray(sobel_x[..., 0] / 4 + 0.5).show()
  ```

  Args:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
      float64.  The image(s) must be 2x2 or larger.

  Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
  """
    static_image_shape = image.get_shape()
    image_shape = array_ops.shape(image)
    kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    num_kernels = len(kernels)
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2)
    kernels_tf = constant_op.constant(kernels, dtype=image.dtype)
    kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')
    pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
    padded = array_ops.pad(image, pad_sizes, mode='REFLECT')
    strides = [1, 1, 1, 1]
    output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
    shape = array_ops.concat([image_shape, [num_kernels]], 0)
    output = array_ops.reshape(output, shape=shape)
    output.set_shape(static_image_shape.concatenate([num_kernels]))
    return output