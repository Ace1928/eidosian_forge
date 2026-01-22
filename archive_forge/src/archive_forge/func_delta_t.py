from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def delta_t(self) -> float:
    """Time increment of STFT.

        The time increment `delta_t` = `T` * `hop` represents the sample
        increment `hop` converted to time based on the sampling interval `T`.

        See Also
        --------
        delta_f: Width of the frequency bins of the STFT.
        hop: Hop size in signal samples for sliding window.
        t: Times of STFT for an input signal with `n` samples.
        T: Sampling interval of input signal and window `win`.
        ShortTimeFFT: Class this property belongs to
        """
    return self.T * self.hop