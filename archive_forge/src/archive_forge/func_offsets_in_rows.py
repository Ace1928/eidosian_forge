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
def offsets_in_rows(self):
    """Return the offset of each value.

    RowPartition takes an array x and converts it into sublists.
    offsets[i] is the index of x[i] in its sublist.
    Given a shape, such as:
    [*,*,*],[*,*],[],[*,*]
    This returns:
    0,1,2,0,1,0,1

    Returns:
      an offset for every value.
    """
    return gen_ragged_math_ops.ragged_range(starts=constant_op.constant(0, self.dtype), limits=self.row_lengths(), deltas=constant_op.constant(1, self.dtype)).rt_dense_values