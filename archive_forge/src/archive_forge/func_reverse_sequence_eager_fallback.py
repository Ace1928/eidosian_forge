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
def reverse_sequence_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_ReverseSequence_T], seq_lengths: _atypes.TensorFuzzingAnnotation[TV_ReverseSequence_Tlen], seq_dim: int, batch_dim: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ReverseSequence_T]:
    seq_dim = _execute.make_int(seq_dim, 'seq_dim')
    if batch_dim is None:
        batch_dim = 0
    batch_dim = _execute.make_int(batch_dim, 'batch_dim')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _attr_Tlen, (seq_lengths,) = _execute.args_to_matching_eager([seq_lengths], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int64)
    _inputs_flat = [input, seq_lengths]
    _attrs = ('seq_dim', seq_dim, 'batch_dim', batch_dim, 'T', _attr_T, 'Tlen', _attr_Tlen)
    _result = _execute.execute(b'ReverseSequence', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ReverseSequence', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result