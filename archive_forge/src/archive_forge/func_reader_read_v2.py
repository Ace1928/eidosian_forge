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
def reader_read_v2(reader_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], queue_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], name=None):
    """Returns the next record (key, value pair) produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    queue_handle: A `Tensor` of type `resource`.
      Handle to a Queue, with string work items.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, value).

    key: A `Tensor` of type `string`.
    value: A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ReaderReadV2', name, reader_handle, queue_handle)
            _result = _ReaderReadV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return reader_read_v2_eager_fallback(reader_handle, queue_handle, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ReaderReadV2', reader_handle=reader_handle, queue_handle=queue_handle, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('ReaderReadV2', _inputs_flat, _attrs, _result)
    _result = _ReaderReadV2Output._make(_result)
    return _result