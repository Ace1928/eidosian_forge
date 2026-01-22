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
def audio_spectrogram(input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], window_size: int, stride: int, magnitude_squared: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Produces a visualization of audio data over time.

  Spectrograms are a standard way of representing audio information as a series of
  slices of frequency information, one slice for each window of time. By joining
  these together into a sequence, they form a distinctive fingerprint of the sound
  over time.

  This op expects to receive audio data as an input, stored as floats in the range
  -1 to 1, together with a window width in samples, and a stride specifying how
  far to move the window between slices. From this it generates a three
  dimensional output. The first dimension is for the channels in the input, so a
  stereo audio input would have two here for example. The second dimension is time,
  with successive frequency slices. The third dimension has an amplitude value for
  each frequency during that time slice.

  This means the layout when converted and saved as an image is rotated 90 degrees
  clockwise from a typical spectrogram. Time is descending down the Y axis, and
  the frequency decreases from left to right.

  Each value in the result represents the square root of the sum of the real and
  imaginary parts of an FFT on the current window of samples. In this way, the
  lowest dimension represents the power of each frequency in the current window,
  and adjacent windows are concatenated in the next dimension.

  To get a more intuitive and visual look at what this operation does, you can run
  tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
  resulting spectrogram as a PNG image.

  Args:
    input: A `Tensor` of type `float32`. Float representation of audio data.
    window_size: An `int`.
      How wide the input window is in samples. For the highest efficiency
      this should be a power of two, but other values are accepted.
    stride: An `int`.
      How widely apart the center of adjacent sample windows should be.
    magnitude_squared: An optional `bool`. Defaults to `False`.
      Whether to return the squared magnitude or just the
      magnitude. Using squared magnitude can avoid extra calculations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AudioSpectrogram', name, input, 'window_size', window_size, 'stride', stride, 'magnitude_squared', magnitude_squared)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return audio_spectrogram_eager_fallback(input, window_size=window_size, stride=stride, magnitude_squared=magnitude_squared, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    window_size = _execute.make_int(window_size, 'window_size')
    stride = _execute.make_int(stride, 'stride')
    if magnitude_squared is None:
        magnitude_squared = False
    magnitude_squared = _execute.make_bool(magnitude_squared, 'magnitude_squared')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AudioSpectrogram', input=input, window_size=window_size, stride=stride, magnitude_squared=magnitude_squared, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('window_size', _op._get_attr_int('window_size'), 'stride', _op._get_attr_int('stride'), 'magnitude_squared', _op._get_attr_bool('magnitude_squared'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('AudioSpectrogram', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result