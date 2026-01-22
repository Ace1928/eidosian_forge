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
@tf_export('audio.decode_wav')
def decode_wav(contents: _atypes.TensorFuzzingAnnotation[_atypes.String], desired_channels: int=-1, desired_samples: int=-1, name=None):
    """Decode a 16-bit PCM WAV file to a float tensor.

  The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.

  When desired_channels is set, if the input contains fewer channels than this
  then the last channel will be duplicated to give the requested number, else if
  the input has more channels than requested then the additional channels will be
  ignored.

  If desired_samples is set, then the audio will be cropped or padded with zeroes
  to the requested length.

  The first output contains a Tensor with the content of the audio samples. The
  lowest dimension will be the number of channels, and the second will be the
  number of samples. For example, a ten-sample-long stereo WAV file should give an
  output shape of [10, 2].

  Args:
    contents: A `Tensor` of type `string`.
      The WAV-encoded audio, usually from a file.
    desired_channels: An optional `int`. Defaults to `-1`.
      Number of sample channels wanted.
    desired_samples: An optional `int`. Defaults to `-1`.
      Length of audio requested.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (audio, sample_rate).

    audio: A `Tensor` of type `float32`.
    sample_rate: A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DecodeWav', name, contents, 'desired_channels', desired_channels, 'desired_samples', desired_samples)
            _result = _DecodeWavOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_decode_wav((contents, desired_channels, desired_samples, name), None)
            if _result is not NotImplemented:
                return _result
            return decode_wav_eager_fallback(contents, desired_channels=desired_channels, desired_samples=desired_samples, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(decode_wav, (), dict(contents=contents, desired_channels=desired_channels, desired_samples=desired_samples, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_decode_wav((contents, desired_channels, desired_samples, name), None)
        if _result is not NotImplemented:
            return _result
    if desired_channels is None:
        desired_channels = -1
    desired_channels = _execute.make_int(desired_channels, 'desired_channels')
    if desired_samples is None:
        desired_samples = -1
    desired_samples = _execute.make_int(desired_samples, 'desired_samples')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('DecodeWav', contents=contents, desired_channels=desired_channels, desired_samples=desired_samples, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(decode_wav, (), dict(contents=contents, desired_channels=desired_channels, desired_samples=desired_samples, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('desired_channels', _op._get_attr_int('desired_channels'), 'desired_samples', _op._get_attr_int('desired_samples'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DecodeWav', _inputs_flat, _attrs, _result)
    _result = _DecodeWavOutput._make(_result)
    return _result