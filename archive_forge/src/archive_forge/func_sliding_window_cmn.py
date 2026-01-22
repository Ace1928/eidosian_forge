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
def sliding_window_cmn(specgram: Tensor, cmn_window: int=600, min_cmn_window: int=100, center: bool=False, norm_vars: bool=False) -> Tensor:
    """
    Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        specgram (Tensor): Tensor of spectrogram of dimension `(..., time, freq)`
        cmn_window (int, optional): Window in frames for running average CMN computation (int, default = 600)
        min_cmn_window (int, optional):  Minimum CMN window used at start of decoding (adds latency only at start).
            Only applicable if center == false, ignored if center==true (int, default = 100)
        center (bool, optional): If true, use a window centered on the current frame
            (to the extent possible, modulo end effects). If false, window is to the left. (bool, default = false)
        norm_vars (bool, optional): If true, normalize variance to one. (bool, default = false)

    Returns:
        Tensor: Tensor matching input shape `(..., freq, time)`
    """
    input_shape = specgram.shape
    num_frames, num_feats = input_shape[-2:]
    specgram = specgram.view(-1, num_frames, num_feats)
    num_channels = specgram.shape[0]
    dtype = specgram.dtype
    device = specgram.device
    last_window_start = last_window_end = -1
    cur_sum = torch.zeros(num_channels, num_feats, dtype=dtype, device=device)
    cur_sumsq = torch.zeros(num_channels, num_feats, dtype=dtype, device=device)
    cmn_specgram = torch.zeros(num_channels, num_frames, num_feats, dtype=dtype, device=device)
    for t in range(num_frames):
        window_start = 0
        window_end = 0
        if center:
            window_start = t - cmn_window // 2
            window_end = window_start + cmn_window
        else:
            window_start = t - cmn_window
            window_end = t + 1
        if window_start < 0:
            window_end -= window_start
            window_start = 0
        if not center:
            if window_end > t:
                window_end = max(t + 1, min_cmn_window)
        if window_end > num_frames:
            window_start -= window_end - num_frames
            window_end = num_frames
            if window_start < 0:
                window_start = 0
        if last_window_start == -1:
            input_part = specgram[:, window_start:window_end - window_start, :]
            cur_sum += torch.sum(input_part, 1)
            if norm_vars:
                cur_sumsq += torch.cumsum(input_part ** 2, 1)[:, -1, :]
        else:
            if window_start > last_window_start:
                frame_to_remove = specgram[:, last_window_start, :]
                cur_sum -= frame_to_remove
                if norm_vars:
                    cur_sumsq -= frame_to_remove ** 2
            if window_end > last_window_end:
                frame_to_add = specgram[:, last_window_end, :]
                cur_sum += frame_to_add
                if norm_vars:
                    cur_sumsq += frame_to_add ** 2
        window_frames = window_end - window_start
        last_window_start = window_start
        last_window_end = window_end
        cmn_specgram[:, t, :] = specgram[:, t, :] - cur_sum / window_frames
        if norm_vars:
            if window_frames == 1:
                cmn_specgram[:, t, :] = torch.zeros(num_channels, num_feats, dtype=dtype, device=device)
            else:
                variance = cur_sumsq
                variance = variance / window_frames
                variance -= cur_sum ** 2 / window_frames ** 2
                variance = torch.pow(variance, -0.5)
                cmn_specgram[:, t, :] *= variance
    cmn_specgram = cmn_specgram.view(input_shape[:-2] + (num_frames, num_feats))
    if len(input_shape) == 2:
        cmn_specgram = cmn_specgram.squeeze(0)
    return cmn_specgram