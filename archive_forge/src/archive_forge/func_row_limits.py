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
def row_limits(self):
    """Returns the limit indices for rows in this row partition.

    These indices specify where the values for each row end.
    `partition.row_limits()` is equal to `partition.row_splits()[:-1]`.

    Returns:
      A 1-D integer Tensor with shape `[self.nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.
      `self.row_limits()[-1] == self.nvals()`.
    """
    return self._row_splits[1:]