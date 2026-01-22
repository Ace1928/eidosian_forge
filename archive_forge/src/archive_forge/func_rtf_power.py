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
def rtf_power(psd_s: Tensor, psd_n: Tensor, reference_channel: Union[int, Tensor], n_iter: int=3, diagonal_loading: bool=True, diag_eps: float=1e-07) -> Tensor:
    """Estimate the relative transfer function (RTF) or the steering vector by the power method.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor with dimensions `(..., freq, channel, channel)`.
        psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
            Tensor with dimensions `(..., freq, channel, channel)`.
        reference_channel (int or torch.Tensor): Specifies the reference channel.
            If the dtype is ``int``, it represents the reference channel index.
            If the dtype is ``torch.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
            is one-hot.
        diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
            (Default: ``True``)
        diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
            It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)

    Returns:
        torch.Tensor: The estimated complex-valued RTF of target speech.
        Tensor of dimension `(..., freq, channel)`.
    """
    _assert_psd_matrices(psd_s, psd_n)
    if n_iter <= 0:
        raise ValueError('The number of iteration must be greater than 0.')
    if diagonal_loading:
        psd_n = _tik_reg(psd_n, reg=diag_eps)
    phi = torch.linalg.solve(psd_n, psd_s)
    if torch.jit.isinstance(reference_channel, int):
        rtf = phi[..., reference_channel]
    elif torch.jit.isinstance(reference_channel, Tensor):
        reference_channel = reference_channel.to(psd_n.dtype)
        rtf = torch.einsum('...c,...c->...', [phi, reference_channel[..., None, None, :]])
    else:
        raise TypeError(f'Expected "int" or "Tensor" for reference_channel. Found: {type(reference_channel)}.')
    rtf = rtf.unsqueeze(-1)
    if n_iter >= 2:
        for _ in range(n_iter - 2):
            rtf = torch.matmul(phi, rtf)
        rtf = torch.matmul(psd_s, rtf)
    else:
        rtf = torch.matmul(psd_n, rtf)
    return rtf.squeeze(-1)