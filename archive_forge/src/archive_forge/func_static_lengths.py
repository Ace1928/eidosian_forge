import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def static_lengths(self, ragged_lengths=True):
    """Returns a list of statically known axis lengths.

    This represents what values are known. For each row partition, it presents
    either the uniform row length (if statically known),
    the list of row lengths, or none if it is not statically known.
    For the inner shape, if the rank is known, then each dimension is reported
    if known, and None otherwise. If the rank of the inner shape is not known,
    then the returned list ends with an ellipsis.

    Args:
      ragged_lengths: If false, returns None for all ragged dimensions.

    Returns:
      A Sequence[Union[Sequence[int],int, None]] of lengths, with a possible
      Ellipsis at the end.
    """
    if self.num_row_partitions == 0:
        return self._static_inner_shape_as_list(False)
    first_dim = self.row_partitions[0].static_nrows
    if isinstance(first_dim, tensor_shape.Dimension):
        first_dim = first_dim.value
    rp_dims = [first_dim]
    for rp in self.row_partitions:
        if rp.is_uniform():
            rp_dims.append(rp.static_uniform_row_length)
        elif ragged_lengths:
            const_vals = tensor_util.constant_value(rp.row_lengths())
            if const_vals is None:
                rp_dims.append(None)
            else:
                rp_dims.append(tuple(const_vals.tolist()))
        else:
            rp_dims.append(None)
    return rp_dims + self._static_inner_shape_as_list(True)