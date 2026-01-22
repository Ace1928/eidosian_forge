import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export('nn.conv1d', v1=[])
@dispatch.add_dispatch_support
def conv1d_v2(input, filters, stride, padding, data_format='NWC', dilations=None, name=None):
    """Computes a 1-D convolution given 3-D input and filter tensors.

  Given an input tensor of shape
    `batch_shape + [in_width, in_channels]`
  if `data_format` is `"NWC"`, or
    `batch_shape + [in_channels, in_width]`
  if `data_format` is `"NCW"`,
  and a filter / kernel tensor of shape
  `[filter_width, in_channels, out_channels]`, this op reshapes
  the arguments to pass them to `conv2d` to perform the equivalent
  convolution operation.

  Internally, this op reshapes the input tensors and invokes `tf.nn.conv2d`.
  For example, if `data_format` does not start with `"NC"`, a tensor of shape
    `batch_shape + [in_width, in_channels]`
  is reshaped to
    `batch_shape + [1, in_width, in_channels]`,
  and the filter is reshaped to
    `[1, filter_width, in_channels, out_channels]`.
  The result is then reshaped back to
    `batch_shape + [out_width, out_channels]`
  \\(where out_width is a function of the stride and padding as in conv2d\\) and
  returned to the caller.

  Args:
    input: A Tensor of rank at least 3. Must be of type `float16`, `float32`, or
      `float64`.
    filters: A Tensor of rank at least 3.  Must have the same type as `input`.
    stride: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: 'SAME' or 'VALID'. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: An optional `string` from `"NWC", "NCW"`.  Defaults to `"NWC"`,
      the data is stored in the order of
      `batch_shape + [in_width, in_channels]`.  The `"NCW"` format stores data
      as `batch_shape + [in_channels, in_width]`.
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      k > 1, there will be k-1 skipped cells between each filter element on that
      dimension. Dilations in the batch and depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.  Has the same type as input.

  Raises:
    ValueError: if `data_format` is invalid.
  """
    return conv1d(input, filters, stride, padding, use_cudnn_on_gpu=True, data_format=data_format, name=name, dilations=dilations)