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
def multi_device_iterator_init(dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], multi_device_iterator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], max_buffer_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    """Initializes the multi device iterator with the given dataset.

  Args:
    dataset: A `Tensor` of type `variant`. Dataset to be iterated upon.
    multi_device_iterator: A `Tensor` of type `resource`.
      A MultiDeviceIteratorResource.
    max_buffer_size: A `Tensor` of type `int64`.
      The maximum size of the host side per device buffer to keep.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MultiDeviceIteratorInit', name, dataset, multi_device_iterator, max_buffer_size)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return multi_device_iterator_init_eager_fallback(dataset, multi_device_iterator, max_buffer_size, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MultiDeviceIteratorInit', dataset=dataset, multi_device_iterator=multi_device_iterator, max_buffer_size=max_buffer_size, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('MultiDeviceIteratorInit', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result