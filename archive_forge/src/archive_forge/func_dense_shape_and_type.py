import abc
import collections
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.linalg.sparse import gen_sparse_csr_matrix_ops as sm_ops
from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import *
def dense_shape_and_type(matrix):
    """Get dense shape and dtype of the tf.Tensor containing the matrix.

  Args:
    matrix: A `tf.Tensor` of type `tf.variant` storing a sparse matrix.

  Returns:
    An instance of `ShapeAndType` with properties `shape` (a `tf.TensorShape`)
    and `dtype` (a `tf.DType`).

  Raises:
    TypeError: if `matrix` is not a tensor or its dtype is not variant.
    ValueError: if `matrix` lacks static handle data containing the dense
      shape and dtype.
  """
    if not isinstance(matrix, tensor_lib.Tensor):
        raise TypeError('matrix should be a tensor, but saw: %s' % (matrix,))
    if matrix.dtype != dtypes.variant:
        raise TypeError('expected matrix to be type tf.variant, but saw: %s' % (matrix.dtype,))
    handle_data = _get_handle_data(matrix)
    if not handle_data or not handle_data.is_set:
        raise ValueError('matrix has missing handle data: %s' % (matrix,))
    if len(handle_data.shape_and_type) != 1:
        raise ValueError("len(matrix.handle_data.shape_and_type) != 1: '%s'" % (handle_data.shape_and_type,))
    return DenseShapeAndType(tensor_shape.TensorShape(handle_data.shape_and_type[0].shape), dtypes.DType(handle_data.shape_and_type[0].dtype))