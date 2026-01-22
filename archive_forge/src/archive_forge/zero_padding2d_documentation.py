import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import conv_utils
from tensorflow.python.util.tf_export import keras_export
Zero-padding layer for 2D input (e.g. picture).

    This layer can add rows and columns of zeros
    at the top, bottom, left and right side of an image tensor.

    Examples:

    >>> input_shape = (1, 1, 2, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[[0 1]
       [2 3]]]]
    >>> y = tf.keras.layers.ZeroPadding2D(padding=1)(x)
    >>> print(y)
    tf.Tensor(
      [[[[0 0]
         [0 0]
         [0 0]
         [0 0]]
        [[0 0]
         [0 1]
         [2 3]
         [0 0]]
        [[0 0]
         [0 0]
         [0 0]
         [0 0]]]], shape=(1, 3, 4, 2), dtype=int64)

    Args:
      padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        - If int: the same symmetric padding
          is applied to height and width.
        - If tuple of 2 ints:
          interpreted as two different
          symmetric padding values for height and width:
          `(symmetric_height_pad, symmetric_width_pad)`.
        - If tuple of 2 tuples of 2 ints:
          interpreted as
          `((top_pad, bottom_pad), (left_pad, right_pad))`
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch_size, channels, height, width)`.
        When unspecified, uses
        `image_data_format` value found in your Keras config file at
         `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to 'channels_last'.

    Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, rows, cols)`

    Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, padded_rows, padded_cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, padded_rows, padded_cols)`
    