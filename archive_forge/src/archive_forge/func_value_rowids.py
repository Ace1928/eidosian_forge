import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def value_rowids(self):
    """Returns the row indices for this row partition.

    `value_rowids` specifies the row index fo reach value.  In particular,
    `value_rowids[i]` is the row index for `values[i]`.

    Returns:
      A 1-D integer `Tensor` with shape `[self.nvals()]`.
      The returned tensor is nonnegative, and is sorted in ascending order.
    """
    if self._value_rowids is not None:
        return self._value_rowids
    return segment_id_ops.row_splits_to_segment_ids(self._row_splits)