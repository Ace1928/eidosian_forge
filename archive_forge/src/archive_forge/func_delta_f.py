from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def delta_f(self) -> float:
    """Width of the frequency bins of the STFT.

        Return the frequency interval `delta_f` = 1 / (`mfft` * `T`).

        See Also
        --------
        delta_t: Time increment of STFT.
        f_pts: Number of points along the frequency axis.
        f: Frequencies values of the STFT.
        mfft: Length of the input for FFT used.
        T: Sampling interval.
        t: Times of STFT for an input signal with `n` samples.
        ShortTimeFFT: Class this property belongs to.
        """
    return 1 / (self.mfft * self.T)