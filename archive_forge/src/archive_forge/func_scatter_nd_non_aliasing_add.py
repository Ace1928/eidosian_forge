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
def scatter_nd_non_aliasing_add(input: _atypes.TensorFuzzingAnnotation[TV_ScatterNdNonAliasingAdd_T], indices: _atypes.TensorFuzzingAnnotation[TV_ScatterNdNonAliasingAdd_Tindices], updates: _atypes.TensorFuzzingAnnotation[TV_ScatterNdNonAliasingAdd_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_ScatterNdNonAliasingAdd_T]:
    """Applies sparse addition to `input` using individual values or slices

  from `updates` according to indices `indices`.  The updates are non-aliasing:
  `input` is only modified in-place if no other operations will use it.
  Otherwise, a copy of `input` is made.  This operation has a gradient with
  respect to both `input` and `updates`.

  `input` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

  `indices` must be integer tensor, containing indices into `input`.
  It must be shape \\\\([d_0, ..., d_{Q-2}, K]\\\\) where `0 < K <= P`.

  The innermost dimension of `indices` (with length `K`) corresponds to
  indices into elements (if `K = P`) or `(P-K)`-dimensional slices
  (if `K < P`) along the `K`th dimension of `input`.

  `updates` is `Tensor` of rank `Q-1+P-K` with shape:

  $$[d_0, ..., d_{Q-2}, input.shape[K], ..., input.shape[P-1]].$$

  For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
  elements. In Python, that addition would look like this:

      input = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      output = tf.scatter_nd_non_aliasing_add(input, indices, updates)
      with tf.Session() as sess:
        print(sess.run(output))

  The resulting value `output` would look like this:

      [1, 13, 3, 14, 14, 6, 7, 20]

  See `tf.scatter_nd` for more details about how to make updates to slices.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      A Tensor.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A Tensor. Must be one of the following types: `int32`, `int64`.
      A tensor of indices into `input`.
    updates: A `Tensor`. Must have the same type as `input`.
      A Tensor. Must have the same type as ref. A tensor of updated values
      to add to `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ScatterNdNonAliasingAdd', name, input, indices, updates)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return scatter_nd_non_aliasing_add_eager_fallback(input, indices, updates, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ScatterNdNonAliasingAdd', input=input, indices=indices, updates=updates, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ScatterNdNonAliasingAdd', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result