import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.gen_sparse_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export
def validate_sparse_weights(values, weights, dtype=None):
    """Validates the passed weight tensor or creates an empty one."""
    if weights is None:
        if dtype:
            return array_ops.constant([], dtype=dtype)
        return array_ops.constant([], dtype=values.values.dtype)
    if not isinstance(weights, sparse_tensor.SparseTensor):
        raise ValueError(f'Argument `weights` must be a SparseTensor if `values` is a SparseTensor. Received weights={weights} of type: {type(weights).__name__}')
    checks = []
    if weights.dense_shape is not values.dense_shape:
        checks.append(check_ops.assert_equal(weights.dense_shape, values.dense_shape, message="'weights' and 'values' must have the same dense shape."))
    if weights.indices is not values.indices:
        checks.append(check_ops.assert_equal(weights.indices, values.indices, message="'weights' and 'values' must have the same indices."))
    if checks:
        with ops.control_dependencies(checks):
            weights = array_ops.identity(weights.values)
    else:
        weights = weights.values
    return weights