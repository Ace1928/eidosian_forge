import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def pitch_shift(waveform: Tensor, sample_rate: int, n_steps: int, bins_per_octave: int=12, n_fft: int=512, win_length: Optional[int]=None, hop_length: Optional[int]=None, window: Optional[Tensor]=None) -> Tensor:
    """
    Shift the pitch of a waveform by ``n_steps`` steps.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        waveform (Tensor): The input waveform of shape `(..., time)`.
        sample_rate (int): Sample rate of `waveform`.
        n_steps (int): The (fractional) steps to shift `waveform`.
        bins_per_octave (int, optional): The number of steps per octave (Default: ``12``).
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins (Default: ``512``).
        win_length (int or None, optional): Window size. If None, then ``n_fft`` is used. (Default: ``None``).
        hop_length (int or None, optional): Length of hop between STFT windows. If None, then
            ``win_length // 4`` is used (Default: ``None``).
        window (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window.
            If None, then ``torch.hann_window(win_length)`` is used (Default: ``None``).


    Returns:
        Tensor: The pitch-shifted audio waveform of shape `(..., time)`.
    """
    waveform_stretch = _stretch_waveform(waveform, n_steps, bins_per_octave, n_fft, win_length, hop_length, window)
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    waveform_shift = resample(waveform_stretch, int(sample_rate / rate), sample_rate)
    return _fix_waveform_shape(waveform_shift, waveform.size())