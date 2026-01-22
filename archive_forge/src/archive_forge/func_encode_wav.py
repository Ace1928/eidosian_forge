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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('audio.encode_wav')
def encode_wav(audio: _atypes.TensorFuzzingAnnotation[_atypes.Float32], sample_rate: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Encode audio data using the WAV file format.

  This operation will generate a string suitable to be saved out to create a .wav
  audio file. It will be encoded in the 16-bit PCM format. It takes in float
  values in the range -1.0f to 1.0f, and any outside that value will be clamped to
  that range.

  `audio` is a 2-D float Tensor of shape `[length, channels]`.
  `sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).

  Args:
    audio: A `Tensor` of type `float32`. 2-D with shape `[length, channels]`.
    sample_rate: A `Tensor` of type `int32`.
      Scalar containing the sample frequency.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'EncodeWav', name, audio, sample_rate)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_encode_wav((audio, sample_rate, name), None)
            if _result is not NotImplemented:
                return _result
            return encode_wav_eager_fallback(audio, sample_rate, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(encode_wav, (), dict(audio=audio, sample_rate=sample_rate, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_encode_wav((audio, sample_rate, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('EncodeWav', audio=audio, sample_rate=sample_rate, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(encode_wav, (), dict(audio=audio, sample_rate=sample_rate, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('EncodeWav', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result