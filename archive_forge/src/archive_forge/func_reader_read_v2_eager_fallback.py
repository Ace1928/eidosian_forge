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
def reader_read_v2_eager_fallback(reader_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], queue_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], name, ctx):
    reader_handle = _ops.convert_to_tensor(reader_handle, _dtypes.resource)
    queue_handle = _ops.convert_to_tensor(queue_handle, _dtypes.resource)
    _inputs_flat = [reader_handle, queue_handle]
    _attrs = None
    _result = _execute.execute(b'ReaderReadV2', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ReaderReadV2', _inputs_flat, _attrs, _result)
    _result = _ReaderReadV2Output._make(_result)
    return _result