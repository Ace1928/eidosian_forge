import warnings
from typing import List, Optional, Union
import torch
from torchaudio.functional import fftconvolve
def oscillator_bank(frequencies: torch.Tensor, amplitudes: torch.Tensor, sample_rate: float, reduction: str='sum', dtype: Optional[torch.dtype]=torch.float64) -> torch.Tensor:
    """Synthesize waveform from the given instantaneous frequencies and amplitudes.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        The phase information of the output waveform is found by taking the cumulative sum
        of the given instantaneous frequencies (``frequencies``).
        This incurs roundoff error when the data type does not have enough precision.
        Using ``torch.float64`` can work around this.

        The following figure shows the difference between ``torch.float32`` and
        ``torch.float64`` when generating a sin wave of constant frequency and amplitude
        with sample rate 8000 [Hz].
        Notice that ``torch.float32`` version shows artifacts that are not seen in
        ``torch.float64`` version.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/oscillator_precision.png

    Args:
        frequencies (Tensor): Sample-wise oscillator frequencies (Hz). Shape `(..., time, N)`.
        amplitudes (Tensor): Sample-wise oscillator amplitude. Shape: `(..., time, N)`.
        sample_rate (float): Sample rate
        reduction (str): Reduction to perform.
            Valid values are ``"sum"``, ``"mean"`` or ``"none"``. Default: ``"sum"``
        dtype (torch.dtype or None, optional): The data type on which cumulative sum operation is performed.
            Default: ``torch.float64``. Pass ``None`` to disable the casting.

    Returns:
        Tensor:
            The resulting waveform.

            If ``reduction`` is ``"none"``, then the shape is
            `(..., time, N)`, otherwise the shape is `(..., time)`.
    """
    if frequencies.shape != amplitudes.shape:
        raise ValueError(f'The shapes of `frequencies` and `amplitudes` must match. Found: {frequencies.shape} and {amplitudes.shape} respectively.')
    reductions = ['sum', 'mean', 'none']
    if reduction not in reductions:
        raise ValueError(f'The value of reduction must be either {reductions}. Found: {reduction}')
    invalid = torch.abs(frequencies) >= sample_rate / 2
    if torch.any(invalid):
        warnings.warn('Some frequencies are above nyquist frequency. Setting the corresponding amplitude to zero. This might cause numerically unstable gradient.')
        amplitudes = torch.where(invalid, 0.0, amplitudes)
    pi2 = 2.0 * torch.pi
    freqs = frequencies * pi2 / sample_rate % pi2
    phases = torch.cumsum(freqs, dim=-2, dtype=dtype)
    if dtype is not None and freqs.dtype != dtype:
        phases = phases.to(freqs.dtype)
    waveform = amplitudes * torch.sin(phases)
    if reduction == 'sum':
        return waveform.sum(-1)
    if reduction == 'mean':
        return waveform.mean(-1)
    return waveform