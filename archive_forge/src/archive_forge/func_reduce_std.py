import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_api(math_ops.reduce_std)
def reduce_std(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=False, name=None):
    """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
    with ops.name_scope(name, 'RaggedReduceStd', [input_tensor, axis]):
        variance = reduce_variance(input_tensor, axis=axis, keepdims=keepdims)
        return math_ops.sqrt(variance)