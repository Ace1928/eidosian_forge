from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@lru_cache(maxsize=256)
def upper_border_begin(self, n: int) -> tuple[int, int]:
    """First signal index and first slice index affected by post-padding.

        Describes the point where the window does begin stick out to the right
        of the signal domain.
        A detailed example is given :ref:`tutorial_stft_sliding_win` section
        of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        ShortTimeFFT: Class this method belongs to.
        """
    w2 = self.win.real ** 2 + self.win.imag ** 2
    q2 = n // self.hop + 1
    q1 = max((n - self.m_num) // self.hop - 1, -1)
    for q_ in range(q2, q1, -1):
        k_ = q_ * self.hop + (self.m_num - self.m_num_mid)
        if k_ < n or all(w2[n - k_:] == 0):
            return ((q_ + 1) * self.hop - self.m_num_mid, q_ + 1)
    return (0, 0)