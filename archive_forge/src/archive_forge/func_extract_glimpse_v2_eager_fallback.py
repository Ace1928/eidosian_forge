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
def extract_glimpse_v2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], offsets: _atypes.TensorFuzzingAnnotation[_atypes.Float32], centered: bool, normalized: bool, uniform_noise: bool, noise: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    if centered is None:
        centered = True
    centered = _execute.make_bool(centered, 'centered')
    if normalized is None:
        normalized = True
    normalized = _execute.make_bool(normalized, 'normalized')
    if uniform_noise is None:
        uniform_noise = True
    uniform_noise = _execute.make_bool(uniform_noise, 'uniform_noise')
    if noise is None:
        noise = 'uniform'
    noise = _execute.make_str(noise, 'noise')
    input = _ops.convert_to_tensor(input, _dtypes.float32)
    size = _ops.convert_to_tensor(size, _dtypes.int32)
    offsets = _ops.convert_to_tensor(offsets, _dtypes.float32)
    _inputs_flat = [input, size, offsets]
    _attrs = ('centered', centered, 'normalized', normalized, 'uniform_noise', uniform_noise, 'noise', noise)
    _result = _execute.execute(b'ExtractGlimpseV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ExtractGlimpseV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result