import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.utils import conv_utils
def make_2d(tensor, split_dim):
    """Reshapes an N-dimensional tensor into a 2D tensor.

    Dimensions before (excluding) and after (including) `split_dim` are grouped
    together.

    Args:
      tensor: a tensor of shape `(d0, ..., d(N-1))`.
      split_dim: an integer from 1 to N-1, index of the dimension to group
        dimensions before (excluding) and after (including).

    Returns:
      Tensor of shape
      `(d0 * ... * d(split_dim-1), d(split_dim) * ... * d(N-1))`.
    """
    shape = tf.shape(tensor)
    in_dims = shape[:split_dim]
    out_dims = shape[split_dim:]
    in_size = tf.reduce_prod(in_dims)
    out_size = tf.reduce_prod(out_dims)
    return tf.reshape(tensor, (in_size, out_size))