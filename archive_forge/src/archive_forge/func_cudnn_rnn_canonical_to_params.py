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
def cudnn_rnn_canonical_to_params(num_layers: _atypes.TensorFuzzingAnnotation[_atypes.Int32], num_units: _atypes.TensorFuzzingAnnotation[_atypes.Int32], input_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], weights: List[_atypes.TensorFuzzingAnnotation[TV_CudnnRNNCanonicalToParams_T]], biases: List[_atypes.TensorFuzzingAnnotation[TV_CudnnRNNCanonicalToParams_T]], rnn_mode: str='lstm', input_mode: str='linear_input', direction: str='unidirectional', dropout: float=0, seed: int=0, seed2: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[TV_CudnnRNNCanonicalToParams_T]:
    """Converts CudnnRNN params from canonical form to usable form.

  Writes a set of weights into the opaque params buffer so they can be used in
  upcoming training or inferences.

  Note that the params buffer may not be compatible across different GPUs. So any
  save and restoration should be converted to and from the canonical weights and
  biases.

  num_layers: Specifies the number of layers in the RNN model.
  num_units: Specifies the size of the hidden state.
  input_size: Specifies the size of the input state.
  weights: the canonical form of weights that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  biases: the canonical form of biases that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  num_params: number of parameter sets for all layers.
      Each layer may contain multiple parameter sets, with each set consisting of
      a weight matrix and a bias vector.
  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
  dropout: dropout probability. When set to 0., dropout is disabled.
  seed: the 1st part of a seed to initialize dropout.
  seed2: the 2nd part of a seed to initialize dropout.

  Args:
    num_layers: A `Tensor` of type `int32`.
    num_units: A `Tensor` of type `int32`.
    input_size: A `Tensor` of type `int32`.
    weights: A list of at least 1 `Tensor` objects with the same type in: `bfloat16`, `half`, `float32`, `float64`.
    biases: A list with the same length as `weights` of `Tensor` objects with the same type as `weights`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CudnnRNNCanonicalToParams', name, num_layers, num_units, input_size, weights, biases, 'rnn_mode', rnn_mode, 'input_mode', input_mode, 'direction', direction, 'dropout', dropout, 'seed', seed, 'seed2', seed2)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return cudnn_rnn_canonical_to_params_eager_fallback(num_layers, num_units, input_size, weights, biases, rnn_mode=rnn_mode, input_mode=input_mode, direction=direction, dropout=dropout, seed=seed, seed2=seed2, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(weights, (list, tuple)):
        raise TypeError("Expected list for 'weights' argument to 'cudnn_rnn_canonical_to_params' Op, not %r." % weights)
    _attr_num_params = len(weights)
    if not isinstance(biases, (list, tuple)):
        raise TypeError("Expected list for 'biases' argument to 'cudnn_rnn_canonical_to_params' Op, not %r." % biases)
    if len(biases) != _attr_num_params:
        raise ValueError("List argument 'biases' to 'cudnn_rnn_canonical_to_params' Op with length %d must match length %d of argument 'weights'." % (len(biases), _attr_num_params))
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CudnnRNNCanonicalToParams', num_layers=num_layers, num_units=num_units, input_size=input_size, weights=weights, biases=biases, rnn_mode=rnn_mode, input_mode=input_mode, direction=direction, dropout=dropout, seed=seed, seed2=seed2, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'num_params', _op._get_attr_int('num_params'), 'rnn_mode', _op.get_attr('rnn_mode'), 'input_mode', _op.get_attr('input_mode'), 'direction', _op.get_attr('direction'), 'dropout', _op.get_attr('dropout'), 'seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CudnnRNNCanonicalToParams', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result