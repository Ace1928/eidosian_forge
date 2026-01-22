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
def quantized_reshape_eager_fallback(tensor: _atypes.TensorFuzzingAnnotation[TV_QuantizedReshape_T], shape: _atypes.TensorFuzzingAnnotation[TV_QuantizedReshape_Tshape], input_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], input_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name, ctx):
    _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
    _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    input_min = _ops.convert_to_tensor(input_min, _dtypes.float32)
    input_max = _ops.convert_to_tensor(input_max, _dtypes.float32)
    _inputs_flat = [tensor, shape, input_min, input_max]
    _attrs = ('T', _attr_T, 'Tshape', _attr_Tshape)
    _result = _execute.execute(b'QuantizedReshape', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedReshape', _inputs_flat, _attrs, _result)
    _result = _QuantizedReshapeOutput._make(_result)
    return _result