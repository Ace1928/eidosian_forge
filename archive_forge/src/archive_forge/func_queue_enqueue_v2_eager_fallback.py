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
def queue_enqueue_v2_eager_fallback(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], components, timeout_ms: int, name, ctx):
    if timeout_ms is None:
        timeout_ms = -1
    timeout_ms = _execute.make_int(timeout_ms, 'timeout_ms')
    _attr_Tcomponents, components = _execute.convert_to_mixed_eager_tensors(components, ctx)
    handle = _ops.convert_to_tensor(handle, _dtypes.resource)
    _inputs_flat = [handle] + list(components)
    _attrs = ('Tcomponents', _attr_Tcomponents, 'timeout_ms', timeout_ms)
    _result = _execute.execute(b'QueueEnqueueV2', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result