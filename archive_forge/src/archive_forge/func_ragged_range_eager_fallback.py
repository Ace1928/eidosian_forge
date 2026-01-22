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
def ragged_range_eager_fallback(starts: _atypes.TensorFuzzingAnnotation[TV_RaggedRange_T], limits: _atypes.TensorFuzzingAnnotation[TV_RaggedRange_T], deltas: _atypes.TensorFuzzingAnnotation[TV_RaggedRange_T], Tsplits: TV_RaggedRange_Tsplits, name, ctx):
    if Tsplits is None:
        Tsplits = _dtypes.int64
    Tsplits = _execute.make_type(Tsplits, 'Tsplits')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([starts, limits, deltas], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64], _dtypes.int32)
    starts, limits, deltas = _inputs_T
    _inputs_flat = [starts, limits, deltas]
    _attrs = ('T', _attr_T, 'Tsplits', Tsplits)
    _result = _execute.execute(b'RaggedRange', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RaggedRange', _inputs_flat, _attrs, _result)
    _result = _RaggedRangeOutput._make(_result)
    return _result