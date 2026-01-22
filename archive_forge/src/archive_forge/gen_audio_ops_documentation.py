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
Transforms a spectrogram into a form that's useful for speech recognition.

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
  