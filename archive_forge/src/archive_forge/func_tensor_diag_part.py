import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('linalg.tensor_diag_part', v1=['linalg.tensor_diag_part', 'diag_part'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('diag_part')
def tensor_diag_part(input, name=None):
    """Returns the diagonal part of the tensor.

  This operation returns a tensor with the `diagonal` part
  of the `input`. The `diagonal` part is computed as follows:

  Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
  tensor of rank `k` with dimensions `[D1,..., Dk]` where:

  `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

  For a rank 2 tensor, `linalg.diag_part` and `linalg.tensor_diag_part`
  produce the same result. For rank 3 and higher, linalg.diag_part extracts
  the diagonal of each inner-most matrix in the tensor. An example where
  they differ is given below.

  >>> x = [[[[1111,1112],[1121,1122]],
  ...       [[1211,1212],[1221,1222]]],
  ...      [[[2111, 2112], [2121, 2122]],
  ...       [[2211, 2212], [2221, 2222]]]
  ...      ]
  >>> tf.linalg.tensor_diag_part(x)
  <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
  array([[1111, 1212],
         [2121, 2222]], dtype=int32)>
  >>> tf.linalg.diag_part(x).shape
  TensorShape([2, 2, 2])

  Args:
    input: A `Tensor` with rank `2k`.
    name: A name for the operation (optional).

  Returns:
    A Tensor containing diagonals of `input`. Has the same type as `input`, and
    rank `k`.
  """
    return gen_array_ops.diag_part(input=input, name=name)