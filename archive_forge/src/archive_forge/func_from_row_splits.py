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
@classmethod
def from_row_splits(cls, row_splits, validate=True, dtype=None, dtype_hint=None):
    """Creates a `RowPartition` with rows partitioned by `row_splits`.

    This `RowPartition` divides a sequence `values` into rows by indicating
    where each row begins and ends:

    ```python
    partitioned_rows = []
    for i in range(len(row_splits) - 1):
      row_start = row_splits[i]
      row_end = row_splits[i + 1]
      partitioned_rows.append(values[row_start:row_end])
    ```

    Args:
      row_splits: A 1-D integer tensor with shape `[nrows+1]`.  Must not be
        empty, and must be sorted in ascending order.  `row_splits[0]` must be
        zero.
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `row_splits`, dtype_hint, or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.

    Returns:
      A `RowPartition`.

    Raises:
      ValueError: If `row_splits` is an empty list.
    """
    if not isinstance(validate, bool):
        raise TypeError('validate must have type bool')
    if isinstance(row_splits, (list, tuple)) and (not row_splits):
        raise ValueError('row_splits tensor may not be empty.')
    if isinstance(row_splits, tensor_lib.TensorSpec):
        return cls(row_splits=row_splits, internal=_row_partition_factory_key)
    with ops.name_scope(None, 'RowPartitionFromRowSplits', [row_splits]):
        row_splits = cls._convert_row_partition(row_splits, 'row_splits', dtype_hint=dtype_hint, dtype=dtype)
        row_splits.shape.assert_has_rank(1)
        if validate:
            msg = 'Arguments to from_row_splits do not form a valid RaggedTensor:'
            checks = [check_ops.assert_rank(row_splits, 1, message=msg + 'rank'), _assert_zero(row_splits[0], message=msg + 'zero'), _assert_monotonic_increasing(row_splits, message=msg + 'monotonic')]
            row_splits = control_flow_ops.with_dependencies(checks, row_splits)
        return cls(row_splits=row_splits, internal=_row_partition_factory_key)