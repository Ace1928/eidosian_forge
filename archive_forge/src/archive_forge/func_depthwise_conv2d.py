import math
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['nn.depthwise_conv2d'])
@dispatch.add_dispatch_support
def depthwise_conv2d(input, filter, strides, padding, rate=None, name=None, data_format=None, dilations=None):
    """Depthwise 2-D convolution.

  Given a 4D input tensor ('NHWC' or 'NCHW' data formats)
  and a filter tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`
  containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
  applies a different filter to each input channel (expanding from 1 channel
  to `channel_multiplier` channels for each), then concatenates the results
  together.  The output has `in_channels * channel_multiplier` channels.

  In detail, with the default NHWC format,

      output[b, i, j, k * channel_multiplier + q] = sum_{di, dj}
           filter[di, dj, k, q] * input[b, strides[1] * i + rate[0] * di,
                                           strides[2] * j + rate[1] * dj, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the
  same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
  If any value in `rate` is greater than 1, we perform atrous depthwise
  convolution, in which case all values in the `strides` tensor must be equal
  to 1.

  Usage Example:

  >>> x = np.array([
  ...     [1., 2.],
  ...     [3., 4.],
  ...     [5., 6.]
  ... ], dtype=np.float32).reshape((1, 3, 2, 1))
  >>> kernel = np.array([
  ...     [1., 2.],
  ...     [3., 4]
  ... ], dtype=np.float32).reshape((2, 1, 1, 2))
  >>> tf.compat.v1.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
  ...                                  padding='VALID').numpy()
    array([[[[10., 14.],
             [14., 20.]],
            [[18., 26.],
             [22., 32.]]]], dtype=float32)

  >>> tf.compat.v1.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
  ...                                  padding=[[0, 0], [1, 0], [1, 0], [0, 0]]
  ...                                 ).numpy()
    array([[[[ 0.,  0.],
             [ 3.,  4.],
             [ 6.,  8.]],
            [[ 0.,  0.],
             [10., 14.],
             [14., 20.]],
            [[ 0.,  0.],
             [18., 26.],
             [22., 32.]]]], dtype=float32)

  Args:
    input: 4-D with shape according to `data_format`.
    filter: 4-D with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
    strides: 1-D of size 4.  The stride of the sliding window for each
      dimension of `input`.
    padding: Controls how to pad the image before applying the convolution. Can
      be the string `"SAME"` or `"VALID"` indicating the type of padding
      algorithm to use, or a list indicating the explicit paddings at the start
      and end of each dimension. When explicit padding is used and data_format
      is `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    rate: 1-D of size 2. The dilation rate in which we sample input values
      across the `height` and `width` dimensions in atrous convolution. If it is
      greater than 1, then all values of strides must be 1.
    name: A name for this operation (optional).
    data_format: The data format for input. Either "NHWC" (default) or "NCHW".
    dilations: Alias of rate.

  Returns:
    A 4-D `Tensor` with shape according to `data_format`.  E.g., for
    "NHWC" format, shape is
    `[batch, out_height, out_width, in_channels * channel_multiplier].`
  """
    rate = deprecated_argument_lookup('dilations', dilations, 'rate', rate)
    with ops.name_scope(name, 'depthwise', [input, filter]) as name:
        input = ops.convert_to_tensor(input, name='tensor_in')
        filter = ops.convert_to_tensor(filter, name='filter_in')
        if rate is None:
            rate = [1, 1]
        if device_context.enclosing_tpu_context() is not None:
            if data_format == 'NCHW':
                dilations = [1, 1, rate[0], rate[1]]
            else:
                dilations = [1, rate[0], rate[1], 1]
            return nn_ops.depthwise_conv2d_native(input=input, filter=filter, strides=strides, padding=padding, data_format=data_format, dilations=dilations, name=name)

        def op(input_converted, _, padding):
            return nn_ops.depthwise_conv2d_native(input=input_converted, filter=filter, strides=strides, padding=padding, data_format=data_format, name=name)
        return nn_ops.with_space_to_batch(input=input, filter_shape=array_ops.shape(filter), dilation_rate=rate, padding=padding, data_format=data_format, op=op)