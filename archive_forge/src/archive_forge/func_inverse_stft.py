import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.ops.signal import reconstruction_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('signal.inverse_stft')
@dispatch.add_dispatch_support
def inverse_stft(stfts, frame_length, frame_step, fft_length=None, window_fn=window_ops.hann_window, name=None):
    """Computes the inverse [Short-time Fourier Transform][stft] of `stfts`.

  To reconstruct an original waveform, a complementary window function should
  be used with `inverse_stft`. Such a window function can be constructed with
  `tf.signal.inverse_stft_window_fn`.
  Example:

  ```python
  frame_length = 400
  frame_step = 160
  waveform = tf.random.normal(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(waveform, frame_length, frame_step)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(frame_step))
  ```

  If a custom `window_fn` is used with `tf.signal.stft`, it must be passed to
  `tf.signal.inverse_stft_window_fn`:

  ```python
  frame_length = 400
  frame_step = 160
  window_fn = tf.signal.hamming_window
  waveform = tf.random.normal(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(
      waveform, frame_length, frame_step, window_fn=window_fn)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(
         frame_step, forward_window_fn=window_fn))
  ```

  Implemented with TPU/GPU-compatible ops and supports gradients.

  Args:
    stfts: A `complex64`/`complex128` `[..., frames, fft_unique_bins]`
      `Tensor` of STFT bins representing a batch of `fft_length`-point STFTs
      where `fft_unique_bins` is `fft_length // 2 + 1`
    frame_length: An integer scalar `Tensor`. The window length in samples.
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    fft_length: An integer scalar `Tensor`. The size of the FFT that produced
      `stfts`. If not provided, uses the smallest power of 2 enclosing
      `frame_length`.
    window_fn: A callable that takes a window length and a `dtype` keyword
      argument and returns a `[window_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, no windowing is used.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `Tensor` of `float32`/`float64` signals representing
    the inverse STFT for each input STFT in `stfts`.

  Raises:
    ValueError: If `stfts` is not at least rank 2, `frame_length` is not scalar,
      `frame_step` is not scalar, or `fft_length` is not scalar.

  [stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
  """
    with ops.name_scope(name, 'inverse_stft', [stfts]):
        stfts = ops.convert_to_tensor(stfts, name='stfts')
        stfts.shape.with_rank_at_least(2)
        frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
        frame_length.shape.assert_has_rank(0)
        frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
        frame_step.shape.assert_has_rank(0)
        if fft_length is None:
            fft_length = _enclosing_power_of_two(frame_length)
        else:
            fft_length = ops.convert_to_tensor(fft_length, name='fft_length')
            fft_length.shape.assert_has_rank(0)
        real_frames = fft_ops.irfft(stfts, [fft_length])
        frame_length_static = tensor_util.constant_value(frame_length)
        if frame_length_static is None or real_frames.shape.ndims is None or real_frames.shape.as_list()[-1] is None:
            real_frames = real_frames[..., :frame_length]
            real_frames_rank = array_ops.rank(real_frames)
            real_frames_shape = array_ops.shape(real_frames)
            paddings = array_ops.concat([array_ops.zeros([real_frames_rank - 1, 2], dtype=frame_length.dtype), [[0, math_ops.maximum(0, frame_length - real_frames_shape[-1])]]], 0)
            real_frames = array_ops.pad(real_frames, paddings)
        elif real_frames.shape.as_list()[-1] > frame_length_static:
            real_frames = real_frames[..., :frame_length_static]
        elif real_frames.shape.as_list()[-1] < frame_length_static:
            pad_amount = frame_length_static - real_frames.shape.as_list()[-1]
            real_frames = array_ops.pad(real_frames, [[0, 0]] * (real_frames.shape.ndims - 1) + [[0, pad_amount]])
        if frame_length_static is not None and real_frames.shape.ndims is not None:
            real_frames.set_shape([None] * (real_frames.shape.ndims - 1) + [frame_length_static])
        if window_fn is not None:
            window = window_fn(frame_length, dtype=stfts.dtype.real_dtype)
            real_frames *= window
        return reconstruction_ops.overlap_and_add(real_frames, frame_step)