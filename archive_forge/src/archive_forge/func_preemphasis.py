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
def preemphasis(waveform, coeff: float=0.97) -> torch.Tensor:
    """Pre-emphasizes a waveform along its last dimension, i.e.
    for each signal :math:`x` in ``waveform``, computes
    output :math:`y` as

    .. math::
        y[i] = x[i] - \\text{coeff} \\cdot x[i - 1]

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Waveform, with shape `(..., N)`.
        coeff (float, optional): Pre-emphasis coefficient. Typically between 0.0 and 1.0.
            (Default: 0.97)

    Returns:
        torch.Tensor: Pre-emphasized waveform, with shape `(..., N)`.
    """
    waveform = waveform.clone()
    waveform[..., 1:] -= coeff * waveform[..., :-1]
    return waveform