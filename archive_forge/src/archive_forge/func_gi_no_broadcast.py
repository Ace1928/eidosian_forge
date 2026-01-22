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
def gi_no_broadcast():
    old_row_starts = array_ops.gather(original_rp.row_splits(), old_value_rowids)
    expected_row_lengths = array_ops.gather(params=original_rp.row_lengths(), indices=bc.gather_index)
    actual_row_lengths = broadcast_rp.row_lengths()
    check_valid = check_ops.assert_equal(expected_row_lengths, actual_row_lengths, message='Cannot broadcast')
    gather_index = old_row_starts + broadcast_rp.offsets_in_rows()
    return control_flow_ops.with_dependencies([check_valid], gather_index)