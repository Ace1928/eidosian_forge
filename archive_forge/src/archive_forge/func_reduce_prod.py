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
@dispatch.dispatch_for_api(math_ops.reduce_prod)
def reduce_prod(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=None, name=None):
    """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
    return ragged_reduce_aggregate(reduce_op=math_ops.reduce_prod, unsorted_segment_op=math_ops.unsorted_segment_prod, rt_input=input_tensor, axis=axis, keepdims=keepdims, name=name or 'RaggedReduceProd')