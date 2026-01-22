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
def decode_wav_eager_fallback(contents: _atypes.TensorFuzzingAnnotation[_atypes.String], desired_channels: int, desired_samples: int, name, ctx):
    if desired_channels is None:
        desired_channels = -1
    desired_channels = _execute.make_int(desired_channels, 'desired_channels')
    if desired_samples is None:
        desired_samples = -1
    desired_samples = _execute.make_int(desired_samples, 'desired_samples')
    contents = _ops.convert_to_tensor(contents, _dtypes.string)
    _inputs_flat = [contents]
    _attrs = ('desired_channels', desired_channels, 'desired_samples', desired_samples)
    _result = _execute.execute(b'DecodeWav', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DecodeWav', _inputs_flat, _attrs, _result)
    _result = _DecodeWavOutput._make(_result)
    return _result