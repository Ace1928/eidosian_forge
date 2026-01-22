import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.utils import conv_utils
def get_locallyconnected_mask(input_shape, kernel_shape, strides, padding, data_format):
    """Return a mask representing connectivity of a locally-connected operation.

    This method returns a masking numpy array of 0s and 1s (of type
    `np.float32`) that, when element-wise multiplied with a fully-connected
    weight tensor, masks out the weights between disconnected input-output pairs
    and thus implements local connectivity through a sparse fully-connected
    weight tensor.

    Assume an unshared convolution with given parameters is applied to an input
    having N spatial dimensions with `input_shape = (d_in1, ..., d_inN)`
    to produce an output with spatial shape `(d_out1, ..., d_outN)` (determined
    by layer parameters such as `strides`).

    This method returns a mask which can be broadcast-multiplied (element-wise)
    with a 2*(N+1)-D weight matrix (equivalent to a fully-connected layer
    between (N+1)-D activations (N spatial + 1 channel dimensions for input and
    output) to make it perform an unshared convolution with given
    `kernel_shape`, `strides`, `padding` and `data_format`.

    Args:
      input_shape: tuple of size N: `(d_in1, ..., d_inN)` spatial shape of the
        input.
      kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
        receptive field.
      strides: tuple of size N, strides along each spatial dimension.
      padding: type of padding, string `"same"` or `"valid"`.
      data_format: a string, `"channels_first"` or `"channels_last"`.

    Returns:
      a `np.float32`-type `np.ndarray` of shape
      `(1, d_in1, ..., d_inN, 1, d_out1, ..., d_outN)`
      if `data_format == `"channels_first"`, or
      `(d_in1, ..., d_inN, 1, d_out1, ..., d_outN, 1)`
      if `data_format == "channels_last"`.

    Raises:
      ValueError: if `data_format` is neither `"channels_first"` nor
                  `"channels_last"`.
    """
    mask = conv_utils.conv_kernel_mask(input_shape=input_shape, kernel_shape=kernel_shape, strides=strides, padding=padding)
    ndims = int(mask.ndim / 2)
    if data_format == 'channels_first':
        mask = np.expand_dims(mask, 0)
        mask = np.expand_dims(mask, -ndims - 1)
    elif data_format == 'channels_last':
        mask = np.expand_dims(mask, ndims)
        mask = np.expand_dims(mask, -1)
    else:
        raise ValueError('Unrecognized data_format: ' + str(data_format))
    return mask