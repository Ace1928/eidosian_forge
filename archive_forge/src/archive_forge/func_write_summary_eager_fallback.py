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
def write_summary_eager_fallback(writer: _atypes.TensorFuzzingAnnotation[_atypes.Resource], step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tensor: _atypes.TensorFuzzingAnnotation[TV_WriteSummary_T], tag: _atypes.TensorFuzzingAnnotation[_atypes.String], summary_metadata: _atypes.TensorFuzzingAnnotation[_atypes.String], name, ctx):
    _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    step = _ops.convert_to_tensor(step, _dtypes.int64)
    tag = _ops.convert_to_tensor(tag, _dtypes.string)
    summary_metadata = _ops.convert_to_tensor(summary_metadata, _dtypes.string)
    _inputs_flat = [writer, step, tensor, tag, summary_metadata]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'WriteSummary', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result