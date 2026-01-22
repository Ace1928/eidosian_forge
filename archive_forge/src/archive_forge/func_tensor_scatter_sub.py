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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tensor_scatter_nd_sub', v1=['tensor_scatter_nd_sub', 'tensor_scatter_sub'])
@deprecated_endpoints('tensor_scatter_sub')
def tensor_scatter_sub(tensor: _atypes.TensorFuzzingAnnotation[TV_TensorScatterSub_T], indices: _atypes.TensorFuzzingAnnotation[TV_TensorScatterSub_Tindices], updates: _atypes.TensorFuzzingAnnotation[TV_TensorScatterSub_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_TensorScatterSub_T]:
    """Subtracts sparse `updates` from an existing tensor according to `indices`.

  This operation creates a new tensor by subtracting sparse `updates` from the
  passed in `tensor`.
  This operation is very similar to `tf.scatter_nd_sub`, except that the updates
  are subtracted from an existing tensor (as opposed to a variable). If the memory
  for the existing tensor cannot be re-used, a copy is made and updated.

  `indices` is an integer tensor containing indices into a new tensor of shape
  `shape`.  The last dimension of `indices` can be at most the rank of `shape`:

      indices.shape[-1] <= shape.rank

  The last dimension of `indices` corresponds to indices into elements
  (if `indices.shape[-1] = shape.rank`) or slices
  (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
  `shape`.  `updates` is a tensor with shape

      indices.shape[:-1] + shape[indices.shape[-1]:]

  The simplest form of tensor_scatter_sub is to subtract individual elements
  from a tensor by index. For example, say we want to insert 4 scattered elements
  in a rank-1 tensor with 8 elements.

  In Python, this scatter subtract operation would look like this:

  ```python
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      tensor = tf.ones([8], dtype=tf.int32)
      updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
      print(updated)
  ```

  The resulting tensor would look like this:

      [1, -10, 1, -9, -8, 1, 1, -11]

  We can also, insert entire slices of a higher rank tensor all at once. For
  example, if we wanted to insert two slices in the first dimension of a
  rank-3 tensor with two matrices of new values.

  In Python, this scatter add operation would look like this:

  ```python
      indices = tf.constant([[0], [2]])
      updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]],
                             [[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]]])
      tensor = tf.ones([4, 4, 4],dtype=tf.int32)
      updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
      print(updated)
  ```

  The resulting tensor would look like this:

      [[[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
       [[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, the index is ignored.

  Args:
    tensor: A `Tensor`. Tensor to copy/update.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorScatterSub', name, tensor, indices, updates)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_tensor_scatter_sub((tensor, indices, updates, name), None)
            if _result is not NotImplemented:
                return _result
            return tensor_scatter_sub_eager_fallback(tensor, indices, updates, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(tensor_scatter_sub, (), dict(tensor=tensor, indices=indices, updates=updates, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_tensor_scatter_sub((tensor, indices, updates, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorScatterSub', tensor=tensor, indices=indices, updates=updates, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(tensor_scatter_sub, (), dict(tensor=tensor, indices=indices, updates=updates, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorScatterSub', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result