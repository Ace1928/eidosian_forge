import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.gen_sparse_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export
@tf_export('sparse.split', v1=[])
def sparse_split_v2(sp_input=None, num_split=None, axis=None, name=None):
    """Split a `SparseTensor` into `num_split` tensors along `axis`.

  If the `sp_input.dense_shape[axis]` is not an integer multiple of `num_split`
  each slice starting from 0:`shape[axis] % num_split` gets extra one
  dimension. For example:

  >>> indices = [[0, 2], [0, 4], [0, 5], [1, 0], [1, 1]]
  >>> values = [1, 2, 3, 4, 5]
  >>> t = tf.sparse.SparseTensor(indices=indices, values=values,
  ...                            dense_shape=[2, 7])
  >>> tf.sparse.to_dense(t)
  <tf.Tensor: shape=(2, 7), dtype=int32, numpy=
  array([[0, 0, 1, 0, 2, 3, 0],
         [4, 5, 0, 0, 0, 0, 0]], dtype=int32)>

  >>> output = tf.sparse.split(sp_input=t, num_split=2, axis=1)
  >>> tf.sparse.to_dense(output[0])
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
  array([[0, 0, 1, 0],
         [4, 5, 0, 0]], dtype=int32)>
  >>> tf.sparse.to_dense(output[1])
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[2, 3, 0],
         [0, 0, 0]], dtype=int32)>

  >>> output = tf.sparse.split(sp_input=t, num_split=2, axis=0)
  >>> tf.sparse.to_dense(output[0])
  <tf.Tensor: shape=(1, 7), dtype=int32, numpy=array([[0, 0, 1, 0, 2, 3, 0]],
  dtype=int32)>
  >>> tf.sparse.to_dense(output[1])
  <tf.Tensor: shape=(1, 7), dtype=int32, numpy=array([[4, 5, 0, 0, 0, 0, 0]],
  dtype=int32)>

  >>> output = tf.sparse.split(sp_input=t, num_split=2, axis=-1)
  >>> tf.sparse.to_dense(output[0])
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
  array([[0, 0, 1, 0],
         [4, 5, 0, 0]], dtype=int32)>
  >>> tf.sparse.to_dense(output[1])
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[2, 3, 0],
         [0, 0, 0]], dtype=int32)>

  Args:
    sp_input: The `SparseTensor` to split.
    num_split: A Python integer. The number of ways to split.
    axis: A 0-D `int32` `Tensor`. The dimension along which to split. Must be in
      range [-rank, rank), where rank is the number of dimensions in the input
      `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    `num_split` `SparseTensor` objects resulting from splitting `value`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
    return sparse_split(sp_input=sp_input, num_split=num_split, axis=axis, name=name, split_dim=None)