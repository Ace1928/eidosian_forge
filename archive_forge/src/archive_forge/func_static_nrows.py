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
@property
def static_nrows(self):
    """The number of rows in this partition, if statically known.

    ```python
    self.row_lengths().shape == [self.static_nrows]
    self.row_starts().shape == [self.static_nrows]
    self.row_limits().shape == [self.static_nrows]
    self.row_splits().shape == [self.static_nrows + 1]
    ```

    Returns:
      The number of rows in this partition as an `int` (if statically known);
      or `None` (otherwise).
    """
    if self._row_splits is not None:
        nrows_plus_one = tensor_shape.dimension_value(self._row_splits.shape[0])
        if nrows_plus_one is not None:
            return nrows_plus_one - 1
    if self._row_lengths is not None:
        nrows = tensor_shape.dimension_value(self._row_lengths.shape[0])
        if nrows is not None:
            return nrows
    if self._nrows is not None:
        return tensor_util.constant_value(self._nrows)
    return None