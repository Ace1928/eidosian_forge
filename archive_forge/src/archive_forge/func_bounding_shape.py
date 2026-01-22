import functools
import operator
import typing
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def bounding_shape(self, axis=None, name=None, out_type=None):
    """Returns the tight bounding box shape for this `RaggedTensor`.

    Args:
      axis: An integer scalar or vector indicating which axes to return the
        bounding box for.  If not specified, then the full bounding box is
        returned.
      name: A name prefix for the returned tensor (optional).
      out_type: `dtype` for the returned tensor.  Defaults to
        `self.row_splits.dtype`.

    Returns:
      An integer `Tensor` (`dtype=self.row_splits.dtype`).  If `axis` is not
      specified, then `output` is a vector with
      `output.shape=[self.shape.ndims]`.  If `axis` is a scalar, then the
      `output` is a scalar.  If `axis` is a vector, then `output` is a vector,
      where `output[i]` is the bounding size for dimension `axis[i]`.

    #### Example:

    >>> rt = tf.ragged.constant([[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]])
    >>> rt.bounding_shape().numpy()
    array([5, 4])

    """
    if out_type is None:
        out_type = self._row_partition.dtype
    else:
        out_type = dtypes.as_dtype(out_type)
    with ops.name_scope(name, 'RaggedBoundingBox', [self, axis]):
        nested_splits = self.nested_row_splits
        rt_flat_values = self.flat_values
        if isinstance(axis, int):
            if axis == 0:
                return array_ops.shape(nested_splits[0], out_type=out_type)[0] - 1
            elif axis == 1:
                result = math_ops.maximum(math_ops.reduce_max(self.row_lengths()), 0)
                if out_type != self._row_partition.dtype:
                    result = math_ops.cast(result, out_type)
                return result
        splits_shape = array_ops.shape(self.row_splits, out_type=out_type)
        flat_values_shape = array_ops.shape(rt_flat_values, out_type=out_type)
        ragged_dimensions = [splits_shape[0] - 1] + [math_ops.maximum(math_ops.reduce_max(splits[1:] - splits[:-1]), 0) for splits in nested_splits]
        inner_dimensions = flat_values_shape[1:]
        if out_type != self._row_partition.dtype:
            ragged_dimensions = [math_ops.cast(d, out_type) for d in ragged_dimensions]
        bbox = array_ops.concat([array_ops_stack.stack(ragged_dimensions), inner_dimensions], axis=0)
        return bbox if axis is None else array_ops.gather(bbox, axis)