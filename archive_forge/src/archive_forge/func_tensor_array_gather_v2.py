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
def tensor_array_gather_v2(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], dtype: TV_TensorArrayGatherV2_dtype, element_shape=None, name=None) -> _atypes.TensorFuzzingAnnotation[TV_TensorArrayGatherV2_dtype]:
    """Deprecated. Use TensorArrayGatherV3

  Args:
    handle: A `Tensor` of type `string`.
    indices: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorArrayGatherV2', name, handle, indices, flow_in, 'dtype', dtype, 'element_shape', element_shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_array_gather_v2_eager_fallback(handle, indices, flow_in, dtype=dtype, element_shape=element_shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    if element_shape is None:
        element_shape = None
    element_shape = _execute.make_shape(element_shape, 'element_shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArrayGatherV2', handle=handle, indices=indices, flow_in=flow_in, dtype=dtype, element_shape=element_shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'element_shape', _op.get_attr('element_shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArrayGatherV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result