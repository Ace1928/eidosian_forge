import itertools
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
def conv_kernel_mask(input_shape, kernel_shape, strides, padding):
    """Compute a mask representing the connectivity of a convolution operation.

  Assume a convolution with given parameters is applied to an input having N
  spatial dimensions with `input_shape = (d_in1, ..., d_inN)` to produce an
  output with shape `(d_out1, ..., d_outN)`. This method returns a boolean array
  of shape `(d_in1, ..., d_inN, d_out1, ..., d_outN)` with `True` entries
  indicating pairs of input and output locations that are connected by a weight.

  Example:

    >>> input_shape = (4,)
    >>> kernel_shape = (2,)
    >>> strides = (1,)
    >>> padding = "valid"
    >>> conv_kernel_mask(input_shape, kernel_shape, strides, padding)
    array([[ True, False, False],
           [ True,  True, False],
           [False,  True,  True],
           [False, False,  True]])

    where rows and columns correspond to inputs and outputs respectively.


  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
      input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
      receptive field.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.
      `"valid"` means no padding. `"same"` results in padding evenly to
      the left/right or up/down of the input such that output has the same
      height/width dimension as the input.

  Returns:
    A boolean 2N-D `np.ndarray` of shape
    `(d_in1, ..., d_inN, d_out1, ..., d_outN)`, where `(d_out1, ..., d_outN)`
    is the spatial shape of the output. `True` entries in the mask represent
    pairs of input-output locations that are connected by a weight.

  Raises:
    ValueError: if `input_shape`, `kernel_shape` and `strides` don't have the
        same number of dimensions.
    NotImplementedError: if `padding` is not in {`"same"`, `"valid"`}.
  """
    if padding not in {'same', 'valid'}:
        raise NotImplementedError('Padding type %s not supported. Only "valid" and "same" are implemented.' % padding)
    in_dims = len(input_shape)
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape,) * in_dims
    if isinstance(strides, int):
        strides = (strides,) * in_dims
    kernel_dims = len(kernel_shape)
    stride_dims = len(strides)
    if kernel_dims != in_dims or stride_dims != in_dims:
        raise ValueError('Number of strides, input and kernel dimensions must all match. Received: %d, %d, %d.' % (stride_dims, in_dims, kernel_dims))
    output_shape = conv_output_shape(input_shape, kernel_shape, strides, padding)
    mask_shape = input_shape + output_shape
    mask = np.zeros(mask_shape, np.bool_)
    output_axes_ticks = [range(dim) for dim in output_shape]
    for output_position in itertools.product(*output_axes_ticks):
        input_axes_ticks = conv_connected_inputs(input_shape, kernel_shape, output_position, strides, padding)
        for input_position in itertools.product(*input_axes_ticks):
            mask[input_position + output_position] = True
    return mask