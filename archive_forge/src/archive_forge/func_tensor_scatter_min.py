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
@tf_export('tensor_scatter_nd_min')
def tensor_scatter_min(tensor: _atypes.TensorFuzzingAnnotation[TV_TensorScatterMin_T], indices: _atypes.TensorFuzzingAnnotation[TV_TensorScatterMin_Tindices], updates: _atypes.TensorFuzzingAnnotation[TV_TensorScatterMin_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_TensorScatterMin_T]:
    """TODO: add doc.

  Args:
    tensor: A `Tensor`. Tensor to update.
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
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorScatterMin', name, tensor, indices, updates)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_tensor_scatter_min((tensor, indices, updates, name), None)
            if _result is not NotImplemented:
                return _result
            return tensor_scatter_min_eager_fallback(tensor, indices, updates, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(tensor_scatter_min, (), dict(tensor=tensor, indices=indices, updates=updates, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_tensor_scatter_min((tensor, indices, updates, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorScatterMin', tensor=tensor, indices=indices, updates=updates, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(tensor_scatter_min, (), dict(tensor=tensor, indices=indices, updates=updates, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorScatterMin', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result