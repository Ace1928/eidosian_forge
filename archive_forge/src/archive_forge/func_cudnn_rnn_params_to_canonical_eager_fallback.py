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
def cudnn_rnn_params_to_canonical_eager_fallback(num_layers: _atypes.TensorFuzzingAnnotation[_atypes.Int32], num_units: _atypes.TensorFuzzingAnnotation[_atypes.Int32], input_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], params: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNParamsToCanonical_T], num_params: int, rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, name, ctx):
    num_params = _execute.make_int(num_params, 'num_params')
    if rnn_mode is None:
        rnn_mode = 'lstm'
    rnn_mode = _execute.make_str(rnn_mode, 'rnn_mode')
    if input_mode is None:
        input_mode = 'linear_input'
    input_mode = _execute.make_str(input_mode, 'input_mode')
    if direction is None:
        direction = 'unidirectional'
    direction = _execute.make_str(direction, 'direction')
    if dropout is None:
        dropout = 0
    dropout = _execute.make_float(dropout, 'dropout')
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    _attr_T, (params,) = _execute.args_to_matching_eager([params], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    num_layers = _ops.convert_to_tensor(num_layers, _dtypes.int32)
    num_units = _ops.convert_to_tensor(num_units, _dtypes.int32)
    input_size = _ops.convert_to_tensor(input_size, _dtypes.int32)
    _inputs_flat = [num_layers, num_units, input_size, params]
    _attrs = ('T', _attr_T, 'num_params', num_params, 'rnn_mode', rnn_mode, 'input_mode', input_mode, 'direction', direction, 'dropout', dropout, 'seed', seed, 'seed2', seed2)
    _result = _execute.execute(b'CudnnRNNParamsToCanonical', num_params + num_params, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CudnnRNNParamsToCanonical', _inputs_flat, _attrs, _result)
    _result = [_result[:num_params]] + _result[num_params:]
    _result = _result[:1] + [_result[1:]]
    _result = _CudnnRNNParamsToCanonicalOutput._make(_result)
    return _result