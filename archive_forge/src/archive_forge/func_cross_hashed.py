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
@tf_export('ragged.cross_hashed')
@dispatch.add_dispatch_support
def cross_hashed(inputs, num_buckets=0, hash_key=None, name=None):
    """Generates hashed feature cross from a list of tensors.

  The input tensors must have `rank=2`, and must all have the same number of
  rows.  The result is a `RaggedTensor` with the same number of rows as the
  inputs, where `result[row]` contains a list of all combinations of values
  formed by taking a single value from each input's corresponding row
  (`inputs[i][row]`).  Values are combined by hashing together their
  fingerprints. E.g.:

  >>> tf.ragged.cross_hashed([tf.ragged.constant([['a'], ['b', 'c']]),
  ...                         tf.ragged.constant([['d'], ['e']]),
  ...                         tf.ragged.constant([['f'], ['g']])],
  ...                        num_buckets=100)
  <tf.RaggedTensor [[78], [66, 74]]>

  Args:
    inputs: A list of `RaggedTensor` or `Tensor` or `SparseTensor`.
    num_buckets: A non-negative `int` that used to bucket the hashed values. If
      `num_buckets != 0`, then `output = hashed_value % num_buckets`.
    hash_key: Integer hash_key that will be used by the `FingerprintCat64`
      function. If not given, a default key is used.
    name: Optional name for the op.

  Returns:
    A 2D `RaggedTensor` of type `int64`.
  """
    return _cross_internal(inputs=inputs, hashed_output=True, num_buckets=num_buckets, hash_key=hash_key, name=name)