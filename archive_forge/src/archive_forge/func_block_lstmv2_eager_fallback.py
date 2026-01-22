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
def block_lstmv2_eager_fallback(seq_len_max: _atypes.TensorFuzzingAnnotation[_atypes.Int64], x: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMV2_T], cs_prev: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMV2_T], h_prev: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMV2_T], w: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMV2_T], wci: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMV2_T], wcf: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMV2_T], wco: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMV2_T], b: _atypes.TensorFuzzingAnnotation[TV_BlockLSTMV2_T], cell_clip: float, use_peephole: bool, name, ctx):
    if cell_clip is None:
        cell_clip = 0
    cell_clip = _execute.make_float(cell_clip, 'cell_clip')
    if use_peephole is None:
        use_peephole = False
    use_peephole = _execute.make_bool(use_peephole, 'use_peephole')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, cs_prev, h_prev, w, wci, wcf, wco, b], ctx, [_dtypes.half, _dtypes.float32])
    x, cs_prev, h_prev, w, wci, wcf, wco, b = _inputs_T
    seq_len_max = _ops.convert_to_tensor(seq_len_max, _dtypes.int64)
    _inputs_flat = [seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b]
    _attrs = ('cell_clip', cell_clip, 'use_peephole', use_peephole, 'T', _attr_T)
    _result = _execute.execute(b'BlockLSTMV2', 7, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BlockLSTMV2', _inputs_flat, _attrs, _result)
    _result = _BlockLSTMV2Output._make(_result)
    return _result