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
@dispatch.dispatch_for_api(math_ops.unsorted_segment_sqrt_n)
def segment_sqrt_n(data: ragged_tensor.RaggedOrDense, segment_ids: ragged_tensor.RaggedOrDense, num_segments, name=None):
    """For docs, see: _RAGGED_SEGMENT_DOCSTRING."""
    with ops.name_scope(name, 'RaggedSegmentSqrtN', [data, segment_ids, num_segments]):
        total = segment_sum(data, segment_ids, num_segments)
        ones = ragged_tensor.RaggedTensor.from_nested_row_splits(array_ops.ones_like(data.flat_values), data.nested_row_splits, validate=False)
        count = segment_sum(ones, segment_ids, num_segments)
        if ragged_tensor.is_ragged(total):
            return total.with_flat_values(total.flat_values / math_ops.sqrt(count.flat_values))
        else:
            return total / math_ops.sqrt(count)