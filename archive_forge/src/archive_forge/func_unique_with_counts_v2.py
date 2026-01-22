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
def unique_with_counts_v2(x: _atypes.TensorFuzzingAnnotation[TV_UniqueWithCountsV2_T], axis: _atypes.TensorFuzzingAnnotation[TV_UniqueWithCountsV2_Taxis], out_idx: TV_UniqueWithCountsV2_out_idx=_dtypes.int32, name=None):
    """Finds unique elements along an axis of a tensor.

  This operation either returns a tensor `y` containing unique elements
  along the `axis` of a tensor. The returned unique elements is sorted
  in the same order as they occur along `axis` in `x`.
  This operation also returns a tensor `idx` and a tensor `count`
  that are the same size as the number of the elements in `x` along the
  `axis` dimension. The `idx` contains the index in the unique output `y`
  and the `count` contains the count in the unique output `y`.
  In other words, for an `1-D` tensor `x` with `axis = None:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```
  x = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8])
  y, idx, count = tf.raw_ops.UniqueWithCountsV2(x=x, axis = [0])
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  count ==> [2, 1, 3, 1, 2]
  ```

  For a `2-D` tensor `x` with `axis = 0`:

  ```
  x = tf.constant([[1, 0, 0],
                  [1, 0, 0],
                  [2, 0, 0]])
  y, idx, count = tf.raw_ops.UniqueWithCountsV2(x=x, axis=[0])
  y ==> [[1, 0, 0],
         [2, 0, 0]]
  idx ==> [0, 0, 1]
  count ==> [2, 1]
  ```

  For a `2-D` tensor `x` with `axis = 1`:

  ```
  x = tf.constant([[1, 0, 0],
                  [1, 0, 0],
                  [2, 0, 0]])
  y, idx, count = tf.raw_ops.UniqueWithCountsV2(x=x, axis=[1])
  y ==> [[1, 0],
         [1, 0],
         [2, 0]]
  idx ==> [0, 1, 1]
  count ==> [1, 2]
  ```

  Args:
    x: A `Tensor`. A `Tensor`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: None). The axis of the Tensor to
      find the unique elements.
    out_idx: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx, count).

    y: A `Tensor`. Has the same type as `x`.
    idx: A `Tensor` of type `out_idx`.
    count: A `Tensor` of type `out_idx`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UniqueWithCountsV2', name, x, axis, 'out_idx', out_idx)
            _result = _UniqueWithCountsV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return unique_with_counts_v2_eager_fallback(x, axis, out_idx=out_idx, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if out_idx is None:
        out_idx = _dtypes.int32
    out_idx = _execute.make_type(out_idx, 'out_idx')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('UniqueWithCountsV2', x=x, axis=axis, out_idx=out_idx, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Taxis', _op._get_attr_type('Taxis'), 'out_idx', _op._get_attr_type('out_idx'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UniqueWithCountsV2', _inputs_flat, _attrs, _result)
    _result = _UniqueWithCountsV2Output._make(_result)
    return _result