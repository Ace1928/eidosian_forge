from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def validate_dense_weights(values, weights, dtype=None):
    """Validates the passed weight tensor or creates an empty one."""
    if weights is None:
        if dtype:
            return array_ops.constant([], dtype=dtype)
        return array_ops.constant([], dtype=values.dtype)
    if not isinstance(weights, tensor.Tensor):
        raise ValueError(f'Argument `weights` must be a tf.Tensor if `values` is a tf.Tensor. Received weights={weights} of type: {type(weights).__name__}')
    return weights