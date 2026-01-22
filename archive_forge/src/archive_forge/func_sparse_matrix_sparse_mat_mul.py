import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def sparse_matrix_sparse_mat_mul(a: _atypes.TensorFuzzingAnnotation[_atypes.Variant], b: _atypes.TensorFuzzingAnnotation[_atypes.Variant], type: TV_SparseMatrixSparseMatMul_type, transpose_a: bool=False, transpose_b: bool=False, adjoint_a: bool=False, adjoint_b: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Sparse-matrix-multiplies two CSR matrices `a` and `b`.

  Performs a matrix multiplication of a sparse matrix `a` with a sparse matrix
  `b`; returns a sparse matrix `a * b`, unless either `a` or `b` is transposed or
  adjointed.

  Each matrix may be transposed or adjointed (conjugated and transposed)
  according to the Boolean parameters `transpose_a`, `adjoint_a`, `transpose_b`
  and `adjoint_b`. At most one of `transpose_a` or `adjoint_a` may be True.
  Similarly, at most one of `transpose_b` or `adjoint_b` may be True.

  The inputs must have compatible shapes. That is, the inner dimension of `a`
  must be equal to the outer dimension of `b`. This requirement is adjusted
  according to whether either `a` or `b` is transposed or adjointed.

  The `type` parameter denotes the type of the matrix elements. Both `a` and `b`
  must have the same type. The supported types are: `float32`, `float64`,
  `complex64` and `complex128`.

  Both `a` and `b` must have the same rank. Broadcasting is not supported. If they
  have rank 3, each batch of 2D CSRSparseMatrices within `a` and `b` must have the
  same dense shape.

  The sparse matrix product may have numeric (non-structural) zeros.
  TODO(anudhyan): Consider adding a boolean attribute to control whether to prune
  zeros.

  Usage example:

  ```python
      from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

      a_indices = np.array([[0, 0], [2, 3], [2, 4], [3, 0]])
      a_values = np.array([1.0, 5.0, -1.0, -2.0], np.float32)
      a_dense_shape = [4, 5]

      b_indices = np.array([[0, 0], [3, 0], [3, 1]])
      b_values = np.array([2.0, 7.0, 8.0], np.float32)
      b_dense_shape = [5, 3]

      with tf.Session() as sess:
        # Define (COO format) Sparse Tensors over Numpy arrays
        a_st = tf.sparse.SparseTensor(a_indices, a_values, a_dense_shape)
        b_st = tf.sparse.SparseTensor(b_indices, b_values, b_dense_shape)

        # Convert SparseTensors to CSR SparseMatrix
        a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a_st.indices, a_st.values, a_st.dense_shape)
        b_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            b_st.indices, b_st.values, b_st.dense_shape)

        # Compute the CSR SparseMatrix matrix multiplication
        c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
            a=a_sm, b=b_sm, type=tf.float32)

        # Convert the CSR SparseMatrix product to a dense Tensor
        c_sm_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            c_sm, tf.float32)
        # Evaluate the dense Tensor value
        c_sm_dense_value = sess.run(c_sm_dense)
  ```

  `c_sm_dense_value` stores the dense matrix product:

  ```
      [[  2.   0.   0.]
       [  0.   0.   0.]
       [ 35.  40.   0.]
       [ -4.   0.   0.]]
  ```

  a: A `CSRSparseMatrix`.
  b: A `CSRSparseMatrix` with the same type and rank as `a`.
  type: The type of both `a` and `b`.
  transpose_a: If True, `a` transposed before multiplication.
  transpose_b: If True, `b` transposed before multiplication.
  adjoint_a: If True, `a` adjointed before multiplication.
  adjoint_b: If True, `b` adjointed before multiplication.

  Args:
    a: A `Tensor` of type `variant`. A CSRSparseMatrix.
    b: A `Tensor` of type `variant`. A CSRSparseMatrix.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    transpose_a: An optional `bool`. Defaults to `False`.
      Indicates whether `a` should be transposed.
    transpose_b: An optional `bool`. Defaults to `False`.
      Indicates whether `b` should be transposed.
    adjoint_a: An optional `bool`. Defaults to `False`.
      Indicates whether `a` should be conjugate-transposed.
    adjoint_b: An optional `bool`. Defaults to `False`.
      Indicates whether `b` should be conjugate-transposed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseMatrixSparseMatMul', name, a, b, 'type', type, 'transpose_a', transpose_a, 'transpose_b', transpose_b, 'adjoint_a', adjoint_a, 'adjoint_b', adjoint_b)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_matrix_sparse_mat_mul_eager_fallback(a, b, type=type, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    type = _execute.make_type(type, 'type')
    if transpose_a is None:
        transpose_a = False
    transpose_a = _execute.make_bool(transpose_a, 'transpose_a')
    if transpose_b is None:
        transpose_b = False
    transpose_b = _execute.make_bool(transpose_b, 'transpose_b')
    if adjoint_a is None:
        adjoint_a = False
    adjoint_a = _execute.make_bool(adjoint_a, 'adjoint_a')
    if adjoint_b is None:
        adjoint_b = False
    adjoint_b = _execute.make_bool(adjoint_b, 'adjoint_b')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseMatrixSparseMatMul', a=a, b=b, type=type, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('type', _op._get_attr_type('type'), 'transpose_a', _op._get_attr_bool('transpose_a'), 'transpose_b', _op._get_attr_bool('transpose_b'), 'adjoint_a', _op._get_attr_bool('adjoint_a'), 'adjoint_b', _op._get_attr_bool('adjoint_b'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseMatrixSparseMatMul', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result