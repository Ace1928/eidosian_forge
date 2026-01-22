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
def mfcc(spectrogram: _atypes.TensorFuzzingAnnotation[_atypes.Float32], sample_rate: _atypes.TensorFuzzingAnnotation[_atypes.Int32], upper_frequency_limit: float=4000, lower_frequency_limit: float=20, filterbank_channel_count: int=40, dct_coefficient_count: int=13, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Transforms a spectrogram into a form that's useful for speech recognition.

  Mel Frequency Cepstral Coefficients are a way of representing audio data that's
  been effective as an input feature for machine learning. They are created by
  taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
  higher frequencies that are less significant to the human ear. They have a long
  history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
  is a good resource to learn more.

  Args:
    spectrogram: A `Tensor` of type `float32`.
      Typically produced by the Spectrogram op, with magnitude_squared
      set to true.
    sample_rate: A `Tensor` of type `int32`.
      How many samples per second the source audio used.
    upper_frequency_limit: An optional `float`. Defaults to `4000`.
      The highest frequency to use when calculating the
      ceptstrum.
    lower_frequency_limit: An optional `float`. Defaults to `20`.
      The lowest frequency to use when calculating the
      ceptstrum.
    filterbank_channel_count: An optional `int`. Defaults to `40`.
      Resolution of the Mel bank used internally.
    dct_coefficient_count: An optional `int`. Defaults to `13`.
      How many output channels to produce per time slice.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Mfcc', name, spectrogram, sample_rate, 'upper_frequency_limit', upper_frequency_limit, 'lower_frequency_limit', lower_frequency_limit, 'filterbank_channel_count', filterbank_channel_count, 'dct_coefficient_count', dct_coefficient_count)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return mfcc_eager_fallback(spectrogram, sample_rate, upper_frequency_limit=upper_frequency_limit, lower_frequency_limit=lower_frequency_limit, filterbank_channel_count=filterbank_channel_count, dct_coefficient_count=dct_coefficient_count, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if upper_frequency_limit is None:
        upper_frequency_limit = 4000
    upper_frequency_limit = _execute.make_float(upper_frequency_limit, 'upper_frequency_limit')
    if lower_frequency_limit is None:
        lower_frequency_limit = 20
    lower_frequency_limit = _execute.make_float(lower_frequency_limit, 'lower_frequency_limit')
    if filterbank_channel_count is None:
        filterbank_channel_count = 40
    filterbank_channel_count = _execute.make_int(filterbank_channel_count, 'filterbank_channel_count')
    if dct_coefficient_count is None:
        dct_coefficient_count = 13
    dct_coefficient_count = _execute.make_int(dct_coefficient_count, 'dct_coefficient_count')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Mfcc', spectrogram=spectrogram, sample_rate=sample_rate, upper_frequency_limit=upper_frequency_limit, lower_frequency_limit=lower_frequency_limit, filterbank_channel_count=filterbank_channel_count, dct_coefficient_count=dct_coefficient_count, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('upper_frequency_limit', _op.get_attr('upper_frequency_limit'), 'lower_frequency_limit', _op.get_attr('lower_frequency_limit'), 'filterbank_channel_count', _op._get_attr_int('filterbank_channel_count'), 'dct_coefficient_count', _op._get_attr_int('dct_coefficient_count'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Mfcc', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result