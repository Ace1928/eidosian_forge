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
@classmethod
def from_lengths(cls, lengths: Sequence[Union[Sequence[int], int]], num_row_partitions=None, dtype=dtypes.int64):
    """Creates a shape with the given lengths and num_row_partitions.

    The lengths can either be a nonnegative int or a list of nonnegative ints.

    If num_row_partitions is None, then the minimal num_row_partitions is used.

    For example, [2, (3, 2)] is the shape of [[0, 0, 0], [0, 0]], and
    [2, 2] is the shape of [[0, 0], [0, 0]]

    This chooses the minimal num_row_partitions required (including zero).

    The following table gives a few examples (where `RP(lengths)` is short
    for `RowPartition.from_lengths(lengths)`):

    For example:
    from_lengths           | row_partitions            | inner_shape
    ---------------------- | --------------------------| -------------
    []                     | []                        | []
    [2, (3, 2)]            | [RP([3, 2])]              | [5]
    [2, 2]                 | []                        | [2, 2]
    [2, (3, 2), 7]         | [RP([3, 2])]              | [5, 7]
    [2, (2, 2), 3]         | [RP([2, 2])]              | [4, 3]
    [2, 2, 3]              | []                        | [2, 2, 3]
    [2, (2, 1), (2, 0, 3)] | [RP(2, 1), RP([2, 0, 3])] | [5]

    If we want the row partitions to end with uniform row partitions, then
    we can set num_row_partitions.

    For example,
    below URP(3, 12) is RowPartition.from_uniform_row_length(3, 12)

    from_lengths   | num_row_partitions | row_partitions           | inner_shape
    ---------------| -------------------|--------------------------|------------
    [2, (3, 2), 2] | 2                  | [RP([3, 2]), URP(2, 10)] | [10]
    [2, 2]         | 1                  | [URP(2, 4)]              | [4]
    [2, 2, 3]      | 0                  | []                       | [2, 2, 3]
    [2, 2, 3]      | 1                  | [URP(2, 4)]              | [4, 3]
    [2, 2, 3]      | 2                  | [URP(2, 4), URP(3, 12)]  | [12]



    Representing the shapes from init():

    from_lengths             | Tensor Example
    ------------------------ | ------------------------------
    `[2, 3]`                 | `[[1, 2, 3], [4, 5, 6]]`
    `[3, (2, 0, 3)]`         | `[[1, 2], [], [3, 4, 5]]`
    `[2, (2, 1), 2]`         | `[[[1, 2], [3, 4]], [[5, 6]]]`
    `[2, (2, 1), (2, 1, 2)]` | `[[[1, 2], [3]], [[4, 5]]]`

    Args:
      lengths: the lengths of sublists along each axis.
      num_row_partitions: the num_row_partitions of the result or None
        indicating the minimum number of row_partitions.
      dtype: the dtype of the shape (tf.int32 or tf.int64).

    Returns:
      a new DynamicRaggedShape
    """
    if not isinstance(lengths, list):
        raise ValueError('lengths should be a list')
    for x in lengths:
        if not _is_int_or_tuple_of_ints(x):
            raise ValueError('element of lengths should be int or tuple of ints: instead %r' % (x,))
    if num_row_partitions is None:
        is_list = [not isinstance(x, int) for x in lengths]
        if any(is_list):
            num_row_partitions = len(is_list) - is_list[-1::-1].index(True) - 1
        else:
            num_row_partitions = 0
    if not isinstance(num_row_partitions, int):
        raise ValueError('num_row_partitions should be an int or None')
    if not lengths:
        if num_row_partitions > 0:
            raise ValueError('num_row_partitions==0 for a scalar shape')
        return DynamicRaggedShape([], [], dtype=dtype)
    if not num_row_partitions < len(lengths):
        raise ValueError('num_row_partitions should be less than `len(lengths)` if shape is not scalar.')
    if num_row_partitions > 0:
        row_partitions, nvals = _to_row_partitions_and_nvals_from_lengths(lengths[:num_row_partitions + 1])
        inner_shape = [nvals] + lengths[num_row_partitions + 1:]
        return DynamicRaggedShape(row_partitions, inner_shape, dtype=dtype)
    else:
        return DynamicRaggedShape([], lengths, dtype=dtype)