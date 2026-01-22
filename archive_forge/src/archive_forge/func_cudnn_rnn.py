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
def cudnn_rnn(input: _atypes.TensorFuzzingAnnotation[TV_CudnnRNN_T], input_h: _atypes.TensorFuzzingAnnotation[TV_CudnnRNN_T], input_c: _atypes.TensorFuzzingAnnotation[TV_CudnnRNN_T], params: _atypes.TensorFuzzingAnnotation[TV_CudnnRNN_T], rnn_mode: str='lstm', input_mode: str='linear_input', direction: str='unidirectional', dropout: float=0, seed: int=0, seed2: int=0, is_training: bool=True, name=None):
    """A RNN backed by cuDNN.

  Computes the RNN from the input and initial states, with respect to the params
  buffer.

  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
    the actual computation before the first layer. 'skip_input' is only allowed
    when input_size == num_units; 'auto_select' implies 'skip_input' when
    input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used. Should be
    "unidirectional" or "bidirectional".
  dropout: Dropout probability. When set to 0., dropout is disabled.
  seed: The 1st part of a seed to initialize dropout.
  seed2: The 2nd part of a seed to initialize dropout.
  input: A 3-D tensor with the shape of [seq_length, batch_size, input_size].
  input_h: A 3-D tensor with the shape of [num_layer * dir, batch_size,
      num_units].
  input_c: For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
  params: A 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
  output: A 3-D tensor with the shape of [seq_length, batch_size,
      dir * num_units].
  output_h: The same shape has input_h.
  output_c: The same shape as input_c for LSTM. An empty tensor for other models.
  is_training: Indicates whether this operation is used for inference or
    training.
  reserve_space: An opaque tensor that can be used in backprop calculation. It
    is only produced if is_training is false.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_h: A `Tensor`. Must have the same type as `input`.
    input_c: A `Tensor`. Must have the same type as `input`.
    params: A `Tensor`. Must have the same type as `input`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    is_training: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_h, output_c, reserve_space).

    output: A `Tensor`. Has the same type as `input`.
    output_h: A `Tensor`. Has the same type as `input`.
    output_c: A `Tensor`. Has the same type as `input`.
    reserve_space: A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CudnnRNN', name, input, input_h, input_c, params, 'rnn_mode', rnn_mode, 'input_mode', input_mode, 'direction', direction, 'dropout', dropout, 'seed', seed, 'seed2', seed2, 'is_training', is_training)
            _result = _CudnnRNNOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return cudnn_rnn_eager_fallback(input, input_h, input_c, params, rnn_mode=rnn_mode, input_mode=input_mode, direction=direction, dropout=dropout, seed=seed, seed2=seed2, is_training=is_training, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    if is_training is None:
        is_training = True
    is_training = _execute.make_bool(is_training, 'is_training')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CudnnRNN', input=input, input_h=input_h, input_c=input_c, params=params, rnn_mode=rnn_mode, input_mode=input_mode, direction=direction, dropout=dropout, seed=seed, seed2=seed2, is_training=is_training, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'rnn_mode', _op.get_attr('rnn_mode'), 'input_mode', _op.get_attr('input_mode'), 'direction', _op.get_attr('direction'), 'dropout', _op.get_attr('dropout'), 'seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'), 'is_training', _op._get_attr_bool('is_training'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CudnnRNN', _inputs_flat, _attrs, _result)
    _result = _CudnnRNNOutput._make(_result)
    return _result