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
@tf_export('ragged.boolean_mask')
@dispatch.add_dispatch_support
def boolean_mask(data, mask, name=None):
    """Applies a boolean mask to `data` without flattening the mask dimensions.

  Returns a potentially ragged tensor that is formed by retaining the elements
  in `data` where the corresponding value in `mask` is `True`.

  * `output[a1...aA, i, b1...bB] = data[a1...aA, j, b1...bB]`

     Where `j` is the `i`th `True` entry of `mask[a1...aA]`.

  Note that `output` preserves the mask dimensions `a1...aA`; this differs
  from `tf.boolean_mask`, which flattens those dimensions.

  Args:
    data: A potentially ragged tensor.
    mask: A potentially ragged boolean tensor.  `mask`'s shape must be a prefix
      of `data`'s shape.  `rank(mask)` must be known statically.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A potentially ragged tensor that is formed by retaining the elements in
    `data` where the corresponding value in `mask` is `True`.

    * `rank(output) = rank(data)`.
    * `output.ragged_rank = max(data.ragged_rank, rank(mask) - 1)`.

  Raises:
    ValueError: if `rank(mask)` is not known statically; or if `mask.shape` is
      not a prefix of `data.shape`.

  #### Examples:

  >>> # Aliases for True & False so data and mask line up.
  >>> T, F = (True, False)

  >>> tf.ragged.boolean_mask(  # Mask a 2D Tensor.
  ...     data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  ...     mask=[[T, F, T], [F, F, F], [T, F, F]]).to_list()
  [[1, 3], [], [7]]

  >>> tf.ragged.boolean_mask(  # Mask a 2D RaggedTensor.
  ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
  ...     tf.ragged.constant([[F, F, T], [F], [T, T]])).to_list()
  [[3], [], [5, 6]]

  >>> tf.ragged.boolean_mask(  # Mask rows of a 2D RaggedTensor.
  ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
  ...     tf.ragged.constant([True, False, True])).to_list()
  [[1, 2, 3], [5, 6]]
  """
    with ops.name_scope(name, 'RaggedMask', [data, mask]):
        data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
        mask = ragged_tensor.convert_to_tensor_or_ragged_tensor(mask, dtypes.bool, name='mask')
        row_splits_dtype, (data, mask) = ragged_tensor.match_row_splits_dtypes(data, mask, return_dtype=True)
        if mask.shape.ndims is None:
            raise ValueError('mask.shape.ndims must be known statically.')
        elif mask.shape.ndims == 0:
            raise ValueError('mask cannot be scalar.')
        if ragged_tensor.is_ragged(mask):
            if not ragged_tensor.is_ragged(data):
                data = ragged_tensor.RaggedTensor.from_tensor(data, ragged_rank=mask.ragged_rank, row_splits_dtype=mask.row_splits.dtype)
            splits_list = [mask.nested_row_splits, data.nested_row_splits[:mask.ragged_rank]]
            with ops.control_dependencies(ragged_util.assert_splits_match(splits_list)):
                splits = []
                while ragged_tensor.is_ragged(mask):
                    if mask.shape.ndims > 2:
                        splits.append(mask.row_splits)
                    else:
                        int_mask = ragged_functional_ops.map_flat_values(math_ops.cast, mask, dtype=row_splits_dtype)
                        masked_row_lengths = ragged_math_ops.reduce_sum(int_mask, axis=1)
                        splits.append(ragged_util.lengths_to_splits(masked_row_lengths))
                    mask = mask.values
                    data = data.values
                masked_values = boolean_mask(data, mask)
                masked_values = ragged_tensor.RaggedTensor.from_nested_row_splits(masked_values, splits, validate=False)
                return masked_values
        elif ragged_tensor.is_ragged(data) and mask.shape.ndims == 1:
            lengths = data.row_lengths()
            masked_lengths = array_ops.boolean_mask(lengths, mask)
            masked_splits = ragged_util.lengths_to_splits(masked_lengths)
            segment_ids = segment_id_ops.row_splits_to_segment_ids(data.row_splits)
            segment_mask = array_ops.gather(mask, segment_ids)
            masked_values = boolean_mask(data.values, segment_mask)
            return ragged_tensor.RaggedTensor.from_row_splits(masked_values, masked_splits, validate=False)
        if ragged_tensor.is_ragged(data):
            mask = ragged_tensor.RaggedTensor.from_tensor(mask, ragged_rank=min(data.ragged_rank, mask.shape.ndims - 1), row_splits_dtype=data.row_splits.dtype)
            return boolean_mask(data, mask)
        else:
            masked_values = array_ops.boolean_mask(data, mask)
            if mask.shape.ndims >= 2:
                masked_lengths = math_ops.count_nonzero(mask, axis=-1, dtype=row_splits_dtype)
                flattened_masked_lengths = array_ops.reshape(masked_lengths, [-1])
                masked_values = ragged_tensor.RaggedTensor.from_row_lengths(masked_values, flattened_masked_lengths, validate=False)
                if mask.shape.ndims > 2:
                    mask_shape = array_ops.shape(mask, out_type=row_splits_dtype)
                    split_size = math_ops.cumprod(mask_shape) + 1
                    for dim in range(mask.shape.ndims - 3, -1, -1):
                        elt_size = mask_shape[dim + 1]
                        masked_splits = math_ops.range(split_size[dim]) * elt_size
                        masked_values = ragged_tensor.RaggedTensor.from_row_splits(masked_values, masked_splits, validate=False)
            return masked_values