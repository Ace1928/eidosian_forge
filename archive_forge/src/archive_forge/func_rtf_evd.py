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
def rtf_evd(psd_s: Tensor) -> Tensor:
    """Estimate the relative transfer function (RTF) or the steering vector by eigenvalue decomposition.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        psd_s (Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor of dimension `(..., freq, channel, channel)`

    Returns:
        Tensor: The estimated complex-valued RTF of target speech.
        Tensor of dimension `(..., freq, channel)`
    """
    if not psd_s.is_complex():
        raise TypeError(f'The type of psd_s must be ``torch.cfloat`` or ``torch.cdouble``. Found {psd_s.dtype}.')
    if psd_s.shape[-1] != psd_s.shape[-2]:
        raise ValueError(f'The last two dimensions of psd_s should be the same. Found {psd_s.shape}.')
    _, v = torch.linalg.eigh(psd_s)
    rtf = v[..., -1]
    return rtf