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
def lstm_block_cell_grad_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], cs_prev: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], h_prev: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], w: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], wci: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], wcf: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], wco: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], b: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], i: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], cs: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], f: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], o: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], ci: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], co: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], cs_grad: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], h_grad: _atypes.TensorFuzzingAnnotation[TV_LSTMBlockCellGrad_T], use_peephole: bool, name, ctx):
    use_peephole = _execute.make_bool(use_peephole, 'use_peephole')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, cs_grad, h_grad], ctx, [_dtypes.half, _dtypes.float32])
    x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, cs_grad, h_grad = _inputs_T
    _inputs_flat = [x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co, cs_grad, h_grad]
    _attrs = ('use_peephole', use_peephole, 'T', _attr_T)
    _result = _execute.execute(b'LSTMBlockCellGrad', 5, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('LSTMBlockCellGrad', _inputs_flat, _attrs, _result)
    _result = _LSTMBlockCellGradOutput._make(_result)
    return _result