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
@tf_export('sparse.expand_dims')
def sparse_expand_dims(sp_input, axis=None, name=None):
    """Returns a tensor with an length 1 axis inserted at index `axis`.

  Given a tensor `input`, this operation inserts a dimension of length 1 at the
  dimension index `axis` of `input`'s shape. The dimension index follows python
  indexing rules: It's zero-based, a negative index it is counted backward
  from the end.

  This operation is useful to:

  * Add an outer "batch" dimension to a single element.
  * Align axes for broadcasting.
  * To add an inner vector length axis to a tensor of scalars.

  For example:

  If you have a sparse tensor with shape `[height, width, depth]`:

  >>> sp = tf.sparse.SparseTensor(indices=[[3,4,1]], values=[7,],
  ...                             dense_shape=[10,10,3])

  You can add an outer `batch` axis by passing `axis=0`:

  >>> tf.sparse.expand_dims(sp, axis=0).shape.as_list()
  [1, 10, 10, 3]

  The new axis location matches Python `list.insert(axis, 1)`:

  >>> tf.sparse.expand_dims(sp, axis=1).shape.as_list()
  [10, 1, 10, 3]

  Following standard python indexing rules, a negative `axis` counts from the
  end so `axis=-1` adds an inner most dimension:

  >>> tf.sparse.expand_dims(sp, axis=-1).shape.as_list()
  [10, 10, 3, 1]

  Note: Unlike `tf.expand_dims` this function includes a default value for the
  `axis`: `-1`. So if `axis is not specified, an inner dimension is added.

  >>> sp.shape.as_list()
  [10, 10, 3]
  >>> tf.sparse.expand_dims(sp).shape.as_list()
  [10, 10, 3, 1]

  This operation requires that `axis` is a valid index for `input.shape`,
  following python indexing rules:

  ```
  -1-tf.rank(input) <= axis <= tf.rank(input)
  ```

  This operation is related to:

  * `tf.expand_dims`, which provides this functionality for dense tensors.
  * `tf.squeeze`, which removes dimensions of size 1, from dense tensors.
  * `tf.sparse.reshape`, which provides more flexible reshaping capability.

  Args:
    sp_input: A `SparseTensor`.
    axis: 0-D (scalar). Specifies the dimension index at which to expand the
      shape of `input`. Must be in the range `[-rank(sp_input) - 1,
      rank(sp_input)]`. Defaults to `-1`.
    name: The name of the output `SparseTensor`.

  Returns:
    A `SparseTensor` with the same data as `sp_input`, but its shape has an
    additional dimension of size 1 added.
  """
    rank = sp_input.dense_shape.get_shape()[0]
    if rank is None:
        rank = array_ops.shape(sp_input.dense_shape)[0]
    axis = -1 if axis is None else axis
    with ops.name_scope(name, default_name='expand_dims', values=[sp_input]):
        if isinstance(axis, compat.integral_types):
            axis = ops.convert_to_tensor(axis, name='axis', dtype=dtypes.int32)
        elif not isinstance(axis, tensor_lib.Tensor):
            raise TypeError('axis must be an integer value in range [-rank(sp_input) - 1, rank(sp_input)]')
        axis = array_ops.where_v2(axis >= 0, axis, axis + rank + 1)
        column_size = array_ops.shape(sp_input.indices)[0]
        new_index = array_ops.zeros([column_size, 1], dtype=dtypes.int64)
        indices_before = array_ops.slice(sp_input.indices, [0, 0], [-1, axis])
        indices_after = array_ops.slice(sp_input.indices, [0, axis], [-1, -1])
        indices = array_ops.concat([indices_before, new_index, indices_after], axis=1)
        shape_before = array_ops.slice(sp_input.dense_shape, [0], [axis])
        shape_after = array_ops.slice(sp_input.dense_shape, [axis], [-1])
        new_shape = ops.convert_to_tensor([1], name='new_shape', dtype=dtypes.int64)
        shape = array_ops.concat([shape_before, new_shape, shape_after], axis=0)
        return sparse_tensor.SparseTensor(indices=indices, values=sp_input.values, dense_shape=shape)