from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def invertible(self) -> bool:
    """Check if STFT is invertible.

        This is achieved by trying to calculate the canonical dual window.

        See Also
        --------
        :meth:`~ShortTimeFFT.istft`: Inverse short-time Fourier transform.
        m_num: Number of samples in window `win` and `dual_win`.
        dual_win: Canonical dual window.
        win: Window for STFT.
        ShortTimeFFT: Class this property belongs to.
        """
    try:
        return len(self.dual_win) > 0
    except ValueError:
        return False