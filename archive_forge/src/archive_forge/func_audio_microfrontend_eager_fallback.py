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
def audio_microfrontend_eager_fallback(audio: _atypes.TensorFuzzingAnnotation[_atypes.Int16], sample_rate: int, window_size: int, window_step: int, num_channels: int, upper_band_limit: float, lower_band_limit: float, smoothing_bits: int, even_smoothing: float, odd_smoothing: float, min_signal_remaining: float, enable_pcan: bool, pcan_strength: float, pcan_offset: float, gain_bits: int, enable_log: bool, scale_shift: int, left_context: int, right_context: int, frame_stride: int, zero_padding: bool, out_scale: int, out_type: TV_AudioMicrofrontend_out_type, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_AudioMicrofrontend_out_type]:
    if sample_rate is None:
        sample_rate = 16000
    sample_rate = _execute.make_int(sample_rate, 'sample_rate')
    if window_size is None:
        window_size = 25
    window_size = _execute.make_int(window_size, 'window_size')
    if window_step is None:
        window_step = 10
    window_step = _execute.make_int(window_step, 'window_step')
    if num_channels is None:
        num_channels = 32
    num_channels = _execute.make_int(num_channels, 'num_channels')
    if upper_band_limit is None:
        upper_band_limit = 7500
    upper_band_limit = _execute.make_float(upper_band_limit, 'upper_band_limit')
    if lower_band_limit is None:
        lower_band_limit = 125
    lower_band_limit = _execute.make_float(lower_band_limit, 'lower_band_limit')
    if smoothing_bits is None:
        smoothing_bits = 10
    smoothing_bits = _execute.make_int(smoothing_bits, 'smoothing_bits')
    if even_smoothing is None:
        even_smoothing = 0.025
    even_smoothing = _execute.make_float(even_smoothing, 'even_smoothing')
    if odd_smoothing is None:
        odd_smoothing = 0.06
    odd_smoothing = _execute.make_float(odd_smoothing, 'odd_smoothing')
    if min_signal_remaining is None:
        min_signal_remaining = 0.05
    min_signal_remaining = _execute.make_float(min_signal_remaining, 'min_signal_remaining')
    if enable_pcan is None:
        enable_pcan = False
    enable_pcan = _execute.make_bool(enable_pcan, 'enable_pcan')
    if pcan_strength is None:
        pcan_strength = 0.95
    pcan_strength = _execute.make_float(pcan_strength, 'pcan_strength')
    if pcan_offset is None:
        pcan_offset = 80
    pcan_offset = _execute.make_float(pcan_offset, 'pcan_offset')
    if gain_bits is None:
        gain_bits = 21
    gain_bits = _execute.make_int(gain_bits, 'gain_bits')
    if enable_log is None:
        enable_log = True
    enable_log = _execute.make_bool(enable_log, 'enable_log')
    if scale_shift is None:
        scale_shift = 6
    scale_shift = _execute.make_int(scale_shift, 'scale_shift')
    if left_context is None:
        left_context = 0
    left_context = _execute.make_int(left_context, 'left_context')
    if right_context is None:
        right_context = 0
    right_context = _execute.make_int(right_context, 'right_context')
    if frame_stride is None:
        frame_stride = 1
    frame_stride = _execute.make_int(frame_stride, 'frame_stride')
    if zero_padding is None:
        zero_padding = False
    zero_padding = _execute.make_bool(zero_padding, 'zero_padding')
    if out_scale is None:
        out_scale = 1
    out_scale = _execute.make_int(out_scale, 'out_scale')
    if out_type is None:
        out_type = _dtypes.uint16
    out_type = _execute.make_type(out_type, 'out_type')
    audio = _ops.convert_to_tensor(audio, _dtypes.int16)
    _inputs_flat = [audio]
    _attrs = ('sample_rate', sample_rate, 'window_size', window_size, 'window_step', window_step, 'num_channels', num_channels, 'upper_band_limit', upper_band_limit, 'lower_band_limit', lower_band_limit, 'smoothing_bits', smoothing_bits, 'even_smoothing', even_smoothing, 'odd_smoothing', odd_smoothing, 'min_signal_remaining', min_signal_remaining, 'enable_pcan', enable_pcan, 'pcan_strength', pcan_strength, 'pcan_offset', pcan_offset, 'gain_bits', gain_bits, 'enable_log', enable_log, 'scale_shift', scale_shift, 'left_context', left_context, 'right_context', right_context, 'frame_stride', frame_stride, 'zero_padding', zero_padding, 'out_scale', out_scale, 'out_type', out_type)
    _result = _execute.execute(b'AudioMicrofrontend', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('AudioMicrofrontend', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result