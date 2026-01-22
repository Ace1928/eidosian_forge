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
@tf_export(v1=['nn.weighted_moments'])
@dispatch.add_dispatch_support
def weighted_moments(x, axes, frequency_weights, name=None, keep_dims=None, keepdims=None):
    """Returns the frequency-weighted mean and variance of `x`.

  Args:
    x: A tensor.
    axes: 1-d tensor of int32 values; these are the axes along which
      to compute mean and variance.
    frequency_weights: A tensor of positive weights which can be
      broadcast with x.
    name: Name used to scope the operation.
    keep_dims: Produce moments with the same dimensionality as the input.
    keepdims: Alias of keep_dims.

  Returns:
    Two tensors: `weighted_mean` and `weighted_variance`.
  """
    keep_dims = deprecated_argument_lookup('keepdims', keepdims, 'keep_dims', keep_dims)
    if keep_dims is None:
        keep_dims = False
    with ops.name_scope(name, 'weighted_moments', [x, frequency_weights, axes]):
        x = ops.convert_to_tensor(x, name='x')
        frequency_weights = ops.convert_to_tensor(frequency_weights, name='frequency_weights')
        needs_cast = x.dtype == dtypes.float16
        if needs_cast:
            x = math_ops.cast(x, dtypes.float32)
        if frequency_weights.dtype != x.dtype:
            frequency_weights = math_ops.cast(frequency_weights, x.dtype)
        weighted_input_sum = math_ops.reduce_sum(frequency_weights * x, axes, name='weighted_input_sum', keepdims=True)
        broadcasted_weights = frequency_weights + array_ops.zeros_like(x)
        sum_of_weights = math_ops.reduce_sum(broadcasted_weights, axes, name='sum_of_weights', keepdims=True)
        weighted_mean = math_ops.div_no_nan(weighted_input_sum, sum_of_weights)
        weighted_distsq = math_ops.reduce_sum(frequency_weights * math_ops.squared_difference(x, weighted_mean), axes, name='weighted_distsq', keepdims=True)
        weighted_variance = math_ops.div_no_nan(weighted_distsq, sum_of_weights)
        if not keep_dims:
            weighted_mean = array_ops.squeeze(weighted_mean, axis=axes)
            weighted_variance = array_ops.squeeze(weighted_variance, axis=axes)
        if needs_cast:
            weighted_mean = math_ops.cast(weighted_mean, dtypes.float16)
            weighted_variance = math_ops.cast(weighted_variance, dtypes.float16)
        return (weighted_mean, weighted_variance)