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
def matrix_diag_v3_eager_fallback(diagonal: _atypes.TensorFuzzingAnnotation[TV_MatrixDiagV3_T], k: _atypes.TensorFuzzingAnnotation[_atypes.Int32], num_rows: _atypes.TensorFuzzingAnnotation[_atypes.Int32], num_cols: _atypes.TensorFuzzingAnnotation[_atypes.Int32], padding_value: _atypes.TensorFuzzingAnnotation[TV_MatrixDiagV3_T], align: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_MatrixDiagV3_T]:
    if align is None:
        align = 'RIGHT_LEFT'
    align = _execute.make_str(align, 'align')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([diagonal, padding_value], ctx, [])
    diagonal, padding_value = _inputs_T
    k = _ops.convert_to_tensor(k, _dtypes.int32)
    num_rows = _ops.convert_to_tensor(num_rows, _dtypes.int32)
    num_cols = _ops.convert_to_tensor(num_cols, _dtypes.int32)
    _inputs_flat = [diagonal, k, num_rows, num_cols, padding_value]
    _attrs = ('T', _attr_T, 'align', align)
    _result = _execute.execute(b'MatrixDiagV3', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MatrixDiagV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result