import warnings
from typing import Optional, Union
import numpy as np
def fram_wave(waveform: np.array, hop_length: int=160, fft_window_size: int=400, center: bool=True):
    """
    In order to compute the short time fourier transform, the waveform needs to be split in overlapping windowed
    segments called `frames`.

    The window length (window_length) defines how much of the signal is contained in each frame, while the hop length
    defines the step between the beginning of each new frame.


    Args:
        waveform (`np.array` of shape `(sample_length,)`):
            The raw waveform which will be split into smaller chunks.
        hop_length (`int`, *optional*, defaults to 160):
            Step between each window of the waveform.
        fft_window_size (`int`, *optional*, defaults to 400):
            Defines the size of the window.
        center (`bool`, defaults to `True`):
            Whether or not to center each frame around the middle of the frame. Centering is done by reflecting the
            waveform on the left and on the right.

    Return:
        framed_waveform (`np.array` of shape `(waveform.shape // hop_length , fft_window_size)`):
            The framed waveforms that can be fed to `np.fft`.
    """
    warnings.warn('The function `fram_wave` is deprecated and will be removed in version 4.31.0 of Transformers', FutureWarning)
    frames = []
    for i in range(0, waveform.shape[0] + 1, hop_length):
        if center:
            half_window = (fft_window_size - 1) // 2 + 1
            start = i - half_window if i > half_window else 0
            end = i + half_window if i < waveform.shape[0] - half_window else waveform.shape[0]
            frame = waveform[start:end]
            if start == 0:
                padd_width = (-i + half_window, 0)
                frame = np.pad(frame, pad_width=padd_width, mode='reflect')
            elif end == waveform.shape[0]:
                padd_width = (0, i - waveform.shape[0] + half_window)
                frame = np.pad(frame, pad_width=padd_width, mode='reflect')
        else:
            frame = waveform[i:i + fft_window_size]
            frame_width = frame.shape[0]
            if frame_width < waveform.shape[0]:
                frame = np.lib.pad(frame, pad_width=(0, fft_window_size - frame_width), mode='constant', constant_values=0)
        frames.append(frame)
    frames = np.stack(frames, 0)
    return frames