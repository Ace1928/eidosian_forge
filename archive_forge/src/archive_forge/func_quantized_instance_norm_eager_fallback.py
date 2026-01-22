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
def quantized_instance_norm_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_QuantizedInstanceNorm_T], x_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], x_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], output_range_given: bool, given_y_min: float, given_y_max: float, variance_epsilon: float, min_separation: float, name, ctx):
    if output_range_given is None:
        output_range_given = False
    output_range_given = _execute.make_bool(output_range_given, 'output_range_given')
    if given_y_min is None:
        given_y_min = 0
    given_y_min = _execute.make_float(given_y_min, 'given_y_min')
    if given_y_max is None:
        given_y_max = 0
    given_y_max = _execute.make_float(given_y_max, 'given_y_max')
    if variance_epsilon is None:
        variance_epsilon = 1e-05
    variance_epsilon = _execute.make_float(variance_epsilon, 'variance_epsilon')
    if min_separation is None:
        min_separation = 0.001
    min_separation = _execute.make_float(min_separation, 'min_separation')
    _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    x_min = _ops.convert_to_tensor(x_min, _dtypes.float32)
    x_max = _ops.convert_to_tensor(x_max, _dtypes.float32)
    _inputs_flat = [x, x_min, x_max]
    _attrs = ('T', _attr_T, 'output_range_given', output_range_given, 'given_y_min', given_y_min, 'given_y_max', given_y_max, 'variance_epsilon', variance_epsilon, 'min_separation', min_separation)
    _result = _execute.execute(b'QuantizedInstanceNorm', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedInstanceNorm', _inputs_flat, _attrs, _result)
    _result = _QuantizedInstanceNormOutput._make(_result)
    return _result