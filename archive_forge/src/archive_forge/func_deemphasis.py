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
def deemphasis(waveform, coeff: float=0.97) -> torch.Tensor:
    """De-emphasizes a waveform along its last dimension.
    Inverse of :meth:`preemphasis`. Concretely, for each signal
    :math:`x` in ``waveform``, computes output :math:`y` as

    .. math::
        y[i] = x[i] + \\text{coeff} \\cdot y[i - 1]

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Waveform, with shape `(..., N)`.
        coeff (float, optional): De-emphasis coefficient. Typically between 0.0 and 1.0.
            (Default: 0.97)

    Returns:
        torch.Tensor: De-emphasized waveform, with shape `(..., N)`.
    """
    a_coeffs = torch.tensor([1.0, -coeff], dtype=waveform.dtype, device=waveform.device)
    b_coeffs = torch.tensor([1.0, 0.0], dtype=waveform.dtype, device=waveform.device)
    return torchaudio.functional.lfilter(waveform, a_coeffs=a_coeffs, b_coeffs=b_coeffs)