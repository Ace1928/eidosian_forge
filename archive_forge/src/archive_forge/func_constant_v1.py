import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['constant'])
def constant_v1(value, dtype=None, shape=None, name='Const', verify_shape=False):
    """Creates a constant tensor.

  The resulting tensor is populated with values of type `dtype`, as
  specified by arguments `value` and (optionally) `shape` (see examples
  below).

  The argument `value` can be a constant value, or a list of values of type
  `dtype`. If `value` is a list, then the length of the list must be less
  than or equal to the number of elements implied by the `shape` argument (if
  specified). In the case where the list length is less than the number of
  elements specified by `shape`, the last element in the list will be used
  to fill the remaining entries.

  The argument `shape` is optional. If present, it specifies the dimensions of
  the resulting tensor. If not present, the shape of `value` is used.

  If the argument `dtype` is not specified, then the type is inferred from
  the type of `value`.

  For example:

  ```python
  # Constant 1-D Tensor populated with value list.
  tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

  # Constant 2-D tensor populated with scalar value -1.
  tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                               [-1. -1. -1.]]
  ```

  `tf.constant` differs from `tf.fill` in a few ways:

  *   `tf.constant` supports arbitrary constants, not just uniform scalar
      Tensors like `tf.fill`.
  *   `tf.constant` creates a `Const` node in the computation graph with the
      exact value at graph construction time. On the other hand, `tf.fill`
      creates an Op in the graph that is expanded at runtime.
  *   Because `tf.constant` only embeds constant values in the graph, it does
      not support dynamic shapes based on other runtime Tensors, whereas
      `tf.fill` does.

  Args:
    value:          A constant value (or list) of output type `dtype`.

    dtype:          The type of the elements of the resulting tensor.

    shape:          Optional dimensions of resulting tensor.

    name:           Optional name for the tensor.

    verify_shape:   Boolean that enables verification of a shape of values.

  Returns:
    A Constant Tensor.

  Raises:
    TypeError: if shape is incorrectly specified or unsupported.
  """
    return _constant_impl(value, dtype, shape, name, verify_shape=verify_shape, allow_broadcast=False)