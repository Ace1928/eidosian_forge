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
def quantized_concat_eager_fallback(concat_dim: _atypes.TensorFuzzingAnnotation[_atypes.Int32], values: List[_atypes.TensorFuzzingAnnotation[TV_QuantizedConcat_T]], input_mins: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], input_maxes: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], name, ctx):
    if not isinstance(values, (list, tuple)):
        raise TypeError("Expected list for 'values' argument to 'quantized_concat' Op, not %r." % values)
    _attr_N = len(values)
    if not isinstance(input_mins, (list, tuple)):
        raise TypeError("Expected list for 'input_mins' argument to 'quantized_concat' Op, not %r." % input_mins)
    if len(input_mins) != _attr_N:
        raise ValueError("List argument 'input_mins' to 'quantized_concat' Op with length %d must match length %d of argument 'values'." % (len(input_mins), _attr_N))
    if not isinstance(input_maxes, (list, tuple)):
        raise TypeError("Expected list for 'input_maxes' argument to 'quantized_concat' Op, not %r." % input_maxes)
    if len(input_maxes) != _attr_N:
        raise ValueError("List argument 'input_maxes' to 'quantized_concat' Op with length %d must match length %d of argument 'values'." % (len(input_maxes), _attr_N))
    _attr_T, values = _execute.args_to_matching_eager(list(values), ctx, [])
    concat_dim = _ops.convert_to_tensor(concat_dim, _dtypes.int32)
    input_mins = _ops.convert_n_to_tensor(input_mins, _dtypes.float32)
    input_maxes = _ops.convert_n_to_tensor(input_maxes, _dtypes.float32)
    _inputs_flat = [concat_dim] + list(values) + list(input_mins) + list(input_maxes)
    _attrs = ('N', _attr_N, 'T', _attr_T)
    _result = _execute.execute(b'QuantizedConcat', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedConcat', _inputs_flat, _attrs, _result)
    _result = _QuantizedConcatOutput._make(_result)
    return _result