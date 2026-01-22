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
@tf_export('sparse.eye')
def sparse_eye(num_rows, num_columns=None, dtype=dtypes.float32, name=None):
    """Creates a two-dimensional sparse tensor with ones along the diagonal.

  Args:
    num_rows: Non-negative integer or `int32` scalar `tensor` giving the number
      of rows in the resulting matrix.
    num_columns: Optional non-negative integer or `int32` scalar `tensor` giving
      the number of columns in the resulting matrix. Defaults to `num_rows`.
    dtype: The type of element in the resulting `Tensor`.
    name: A name for this `Op`. Defaults to "eye".

  Returns:
    A `SparseTensor` of shape [num_rows, num_columns] with ones along the
    diagonal.
  """
    with ops.name_scope(name, default_name='eye', values=[num_rows, num_columns]):
        num_rows = _make_int64_tensor(num_rows, 'num_rows')
        num_columns = num_rows if num_columns is None else _make_int64_tensor(num_columns, 'num_columns')
        diag_size = math_ops.minimum(num_rows, num_columns)
        diag_range = math_ops.range(diag_size, dtype=dtypes.int64)
        return sparse_tensor.SparseTensor(indices=array_ops_stack.stack([diag_range, diag_range], axis=1), values=array_ops.ones(diag_size, dtype=dtype), dense_shape=[num_rows, num_columns])