import tensorflow.compat.v2 as tf
from keras.src.engine import base_layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Unit normalization layer.

    Normalize a batch of inputs so that each input in the batch has a L2 norm
    equal to 1 (across the axes specified in `axis`).

    Example:

    >>> data = tf.constant(np.arange(6).reshape(2, 3), dtype=tf.float32)
    >>> normalized_data = tf.keras.layers.UnitNormalization()(data)
    >>> print(tf.reduce_sum(normalized_data[0, :] ** 2).numpy())
    1.0

    Args:
      axis: Integer or list/tuple. The axis or axes to normalize across.
        Typically, this is the features axis or axes. The left-out axes are
        typically the batch axis or axes. `-1` is the last dimension
        in the input. Defaults to `-1`.
    