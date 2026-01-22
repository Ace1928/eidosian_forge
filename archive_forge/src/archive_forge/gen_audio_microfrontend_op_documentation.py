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
Audio Microfrontend Op.

  This Op converts a sequence of audio data into one or more
  feature vectors containing filterbanks of the input. The
  conversion process uses a lightweight library to perform:

  1. A slicing window function
  2. Short-time FFTs
  3. Filterbank calculations
  4. Noise reduction
  5. PCAN Auto Gain Control
  6. Logarithmic scaling

  Arguments
    audio: 1D Tensor, int16 audio data in temporal ordering.
    sample_rate: Integer, the sample rate of the audio in Hz.
    window_size: Integer, length of desired time frames in ms.
    window_step: Integer, length of step size for the next frame in ms.
    num_channels: Integer, the number of filterbank channels to use.
    upper_band_limit: Float, the highest frequency included in the filterbanks.
    lower_band_limit: Float, the lowest frequency included in the filterbanks.
    smoothing_bits: Int, scale up signal by 2^(smoothing_bits) before reduction.
    even_smoothing: Float, smoothing coefficient for even-numbered channels.
    odd_smoothing: Float, smoothing coefficient for odd-numbered channels.
    min_signal_remaining: Float, fraction of signal to preserve in smoothing.
    enable_pcan: Bool, enable PCAN auto gain control.
    pcan_strength: Float, gain normalization exponent.
    pcan_offset: Float, positive value added in the normalization denominator.
    gain_bits: Int, number of fractional bits in the gain.
    enable_log: Bool, enable logarithmic scaling of filterbanks.
    scale_shift: Integer, scale filterbanks by 2^(scale_shift).
    left_context: Integer, number of preceding frames to attach to each frame.
    right_context: Integer, number of preceding frames to attach to each frame.
    frame_stride: Integer, M frames to skip over, where output[n] = frame[n*M].
    zero_padding: Bool, if left/right context is out-of-bounds, attach frame of
                  zeroes. Otherwise, frame[0] or frame[size-1] will be copied.
    out_scale: Integer, divide all filterbanks by this number.
    out_type: DType, type of the output Tensor, defaults to UINT16.

  Returns
    filterbanks: 2D Tensor, each row is a time frame, each column is a channel.

  Args:
    audio: A `Tensor` of type `int16`.
    sample_rate: An optional `int`. Defaults to `16000`.
    window_size: An optional `int`. Defaults to `25`.
    window_step: An optional `int`. Defaults to `10`.
    num_channels: An optional `int`. Defaults to `32`.
    upper_band_limit: An optional `float`. Defaults to `7500`.
    lower_band_limit: An optional `float`. Defaults to `125`.
    smoothing_bits: An optional `int`. Defaults to `10`.
    even_smoothing: An optional `float`. Defaults to `0.025`.
    odd_smoothing: An optional `float`. Defaults to `0.06`.
    min_signal_remaining: An optional `float`. Defaults to `0.05`.
    enable_pcan: An optional `bool`. Defaults to `False`.
    pcan_strength: An optional `float`. Defaults to `0.95`.
    pcan_offset: An optional `float`. Defaults to `80`.
    gain_bits: An optional `int`. Defaults to `21`.
    enable_log: An optional `bool`. Defaults to `True`.
    scale_shift: An optional `int`. Defaults to `6`.
    left_context: An optional `int`. Defaults to `0`.
    right_context: An optional `int`. Defaults to `0`.
    frame_stride: An optional `int`. Defaults to `1`.
    zero_padding: An optional `bool`. Defaults to `False`.
    out_scale: An optional `int`. Defaults to `1`.
    out_type: An optional `tf.DType` from: `tf.uint16, tf.float32`. Defaults to `tf.uint16`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  