from typing import Optional
from typing import Union
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def fill_empty_rows(ragged_input, default_value, name=None):
    """Fills empty rows in the input `RaggedTensor` with rank 2 with a default

  value.

  This op adds entries with the specified `default_value` for any row in the
  input that does not already have a value.

  The op also returns an indicator vector such that

      empty_row_indicator[i] = True iff row i was an empty row.

  Args:
    ragged_input: A `RaggedTensor` with rank 2.
    default_value: The value to fill for empty rows, with the same type as
      `ragged_input.`
    name: A name prefix for the returned tensors (optional)

  Returns:
    ragged_ordered_output: A `RaggedTensor`with all empty rows filled in with
      `default_value`.
    empty_row_indicator: A bool vector indicating whether each input row was
      empty.

  Raises:
    TypeError: If `ragged_input` is not a `RaggedTensor`.
  """
    with ops.name_scope(name, 'RaggedFillEmptyRows', [ragged_input]):
        if not isinstance(ragged_input, ragged_tensor.RaggedTensor):
            raise TypeError(f'ragged_input must be RaggedTensor,             got {type(ragged_input)}')
        default_value = ops.convert_to_tensor(default_value, dtype=ragged_input.dtype)
        output_value_rowids, output_values, empty_row_indicator, unused_reverse_index_map = gen_ragged_array_ops.ragged_fill_empty_rows(value_rowids=ragged_input.value_rowids(), values=ragged_input.values, nrows=ragged_input.nrows(), default_value=default_value)
        return (ragged_tensor.RaggedTensor.from_value_rowids(values=output_values, value_rowids=output_value_rowids, validate=False), empty_row_indicator)