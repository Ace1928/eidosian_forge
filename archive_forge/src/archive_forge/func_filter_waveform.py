import warnings
from typing import List, Optional, Union
import torch
from torchaudio.functional import fftconvolve
def filter_waveform(waveform: torch.Tensor, kernels: torch.Tensor, delay_compensation: int=-1):
    """Applies filters along time axis of the given waveform.

    This function applies the given filters along time axis in the following manner:

    1. Split the given waveform into chunks. The number of chunks is equal to the number of given filters.
    2. Filter each chunk with corresponding filter.
    3. Place the filtered chunks at the original indices while adding up the overlapping parts.
    4. Crop the resulting waveform so that delay introduced by the filter is removed and its length
       matches that of the input waveform.

    The following figure illustrates this.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/filter_waveform.png

    .. note::

       If the number of filters is one, then the operation becomes stationary.
       i.e. the same filtering is applied across the time axis.

    Args:
        waveform (Tensor): Shape `(..., time)`.
        kernels (Tensor): Impulse responses.
            Valid inputs are 2D tensor with shape `(num_filters, filter_length)` or
            `(N+1)`-D tensor with shape `(..., num_filters, filter_length)`, where `N` is
            the dimension of waveform.

            In case of 2D input, the same set of filters is used across channels and batches.
            Otherwise, different sets of filters are applied. In this case, the shape of
            the first `N-1` dimensions of filters must match (or be broadcastable to) that of waveform.

        delay_compensation (int): Control how the waveform is cropped after full convolution.
            If the value is zero or positive, it is interpreted as the length of crop at the
            beginning of the waveform. The value cannot be larger than the size of filter kernel.
            Otherwise the initial crop is ``filter_size // 2``.
            When cropping happens, the waveform is also cropped from the end so that the
            length of the resulting waveform matches the input waveform.

    Returns:
        Tensor: `(..., time)`.
    """
    if kernels.ndim not in [2, waveform.ndim + 1]:
        raise ValueError(f'`kernels` must be 2 or N+1 dimension where N is the dimension of waveform. Found: {kernels.ndim} (N={waveform.ndim})')
    num_filters, filter_size = kernels.shape[-2:]
    num_frames = waveform.size(-1)
    if delay_compensation > filter_size:
        raise ValueError(f'When `delay_compenstation` is provided, it cannot be larger than the size of filters.Found: delay_compensation={delay_compensation}, filter_size={filter_size}')
    chunk_length = num_frames // num_filters
    if num_frames % num_filters > 0:
        chunk_length += 1
        num_pad = chunk_length * num_filters - num_frames
        waveform = torch.nn.functional.pad(waveform, [0, num_pad], 'constant', 0)
    chunked = waveform.unfold(-1, chunk_length, chunk_length)
    assert chunked.numel() >= waveform.numel()
    if waveform.ndim + 1 > kernels.ndim:
        expand_shape = waveform.shape[:-1] + kernels.shape
        kernels = kernels.expand(expand_shape)
    convolved = fftconvolve(chunked, kernels)
    restored = _overlap_and_add(convolved, chunk_length)
    if delay_compensation >= 0:
        start = delay_compensation
    else:
        start = filter_size // 2
    num_crops = restored.size(-1) - num_frames
    end = num_crops - start
    result = restored[..., start:-end]
    return result