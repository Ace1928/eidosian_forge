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
@tf_export('scatter_nd', v1=['scatter_nd', 'manip.scatter_nd'])
@deprecated_endpoints('manip.scatter_nd')
def scatter_nd(indices: _atypes.TensorFuzzingAnnotation[TV_ScatterNd_Tindices], updates: _atypes.TensorFuzzingAnnotation[TV_ScatterNd_T], shape: _atypes.TensorFuzzingAnnotation[TV_ScatterNd_Tindices], name=None) -> _atypes.TensorFuzzingAnnotation[TV_ScatterNd_T]:
    """Scatters `updates` into a tensor of shape `shape` according to `indices`.

  Scatter sparse `updates` according to individual values at the specified
  `indices`. This op returns an output tensor with the `shape` you specify. This
  op is the inverse of the `tf.gather_nd` operator which extracts values or slices
  from a given tensor.

  This operation is similar to `tf.tensor_scatter_nd_add`, except that the tensor
  is zero-initialized. Calling `tf.scatter_nd(indices, updates, shape)`
  is identical to calling
  `tf.tensor_scatter_nd_add(tf.zeros(shape, updates.dtype), indices, updates)`

  If `indices` contains duplicates, the associated `updates` are accumulated
  (summed) into the output tensor.

  **WARNING**: For floating-point data types, the output may be nondeterministic.
  This is because the order in which the updates are applied is nondeterministic
  and when floating-point numbers are added in different orders the resulting
  numerical approximation error can be slightly different. However, the output
  will be deterministic if op determinism is enabled via
  `tf.config.experimental.enable_op_determinism`.

  `indices` is an integer tensor containing indices into the output tensor. The
  last dimension of `indices` can be at most the rank of `shape`:

      indices.shape[-1] <= shape.rank

  The last dimension of `indices` corresponds to indices of elements
  (if `indices.shape[-1] = shape.rank`) or slices
  (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
  `shape`.

  `updates` is a tensor with shape:

      indices.shape[:-1] + shape[indices.shape[-1]:]

  The simplest form of the scatter op is to insert individual elements in
  a tensor by index. Consider an example where you want to insert 4 scattered
  elements in a rank-1 tensor with 8 elements.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
  </div>

  In Python, this scatter operation would look like this:

  ```python
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      shape = tf.constant([8])
      scatter = tf.scatter_nd(indices, updates, shape)
      print(scatter)
  ```

  The resulting tensor would look like this:

      [0, 11, 0, 10, 9, 0, 0, 12]

  You can also insert entire slices of a higher rank tensor all at once. For
  example, you can insert two slices in the first dimension of a rank-3 tensor
  with two matrices of new values.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
  </div>

  In Python, this scatter operation would look like this:

  ```python
      indices = tf.constant([[1], [3]])
      updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]],
                             [[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]]])
      shape = tf.constant([4, 4, 4])
      scatter = tf.scatter_nd(indices, updates, shape)
      print(scatter)
  ```

  The resulting tensor would look like this:

      [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
       [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
       [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
       [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, the index is ignored.

  Args:
    indices: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
      Tensor of indices.
    updates: A `Tensor`. Values to scatter into the output tensor.
    shape: A `Tensor`. Must have the same type as `indices`.
      1-D. The shape of the output tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `updates`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ScatterNd', name, indices, updates, shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_scatter_nd((indices, updates, shape, name), None)
            if _result is not NotImplemented:
                return _result
            return scatter_nd_eager_fallback(indices, updates, shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(scatter_nd, (), dict(indices=indices, updates=updates, shape=shape, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_scatter_nd((indices, updates, shape, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('ScatterNd', indices=indices, updates=updates, shape=shape, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(scatter_nd, (), dict(indices=indices, updates=updates, shape=shape, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ScatterNd', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result