import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def riaa_biquad(waveform: Tensor, sample_rate: int) -> Tensor:
    """Apply RIAA vinyl playback equalization.  Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz).
            Allowed sample rates in Hz : ``44100``,``48000``,``88200``,``96000``

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    Reference:
        - http://sox.sourceforge.net/sox.html
        - https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    if sample_rate == 44100:
        zeros = [-0.2014898, 0.923382]
        poles = [0.7083149, 0.9924091]
    elif sample_rate == 48000:
        zeros = [-0.1766069, 0.932159]
        poles = [0.7396325, 0.993133]
    elif sample_rate == 88200:
        zeros = [-0.1168735, 0.9648312]
        poles = [0.8590646, 0.9964002]
    elif sample_rate == 96000:
        zeros = [-0.1141486, 0.9676817]
        poles = [0.8699137, 0.9966946]
    else:
        raise ValueError('Sample rate must be 44.1k, 48k, 88.2k, or 96k')
    b0 = 1.0
    b1 = -(zeros[0] + zeros[1])
    b2 = zeros[0] * zeros[1]
    a0 = 1.0
    a1 = -(poles[0] + poles[1])
    a2 = poles[0] * poles[1]
    y = 2 * math.pi * 1000 / sample_rate
    b_re = b0 + b1 * math.cos(-y) + b2 * math.cos(-2 * y)
    a_re = a0 + a1 * math.cos(-y) + a2 * math.cos(-2 * y)
    b_im = b1 * math.sin(-y) + b2 * math.sin(-2 * y)
    a_im = a1 * math.sin(-y) + a2 * math.sin(-2 * y)
    g = 1 / math.sqrt((b_re ** 2 + b_im ** 2) / (a_re ** 2 + a_im ** 2))
    b0 *= g
    b1 *= g
    b2 *= g
    return biquad(waveform, b0, b1, b2, a0, a1, a2)