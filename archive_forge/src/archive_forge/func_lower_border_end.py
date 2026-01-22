from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def lower_border_end(self) -> tuple[int, int]:
    """First signal index and first slice index unaffected by pre-padding.

        Describes the point where the window does not stick out to the left
        of the signal domain.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this property belongs to.
        """
    if self._lower_border_end is not None:
        return self._lower_border_end
    m0 = np.flatnonzero(self.win.real ** 2 + self.win.imag ** 2)[0]
    k0 = -self.m_num_mid + m0
    for q_, k_ in enumerate(range(k0, self.hop + 1, self.hop)):
        if k_ + self.hop >= 0:
            self._lower_border_end = (k_ + self.m_num, q_ + 1)
            return self._lower_border_end
    self._lower_border_end = (0, max(self.p_min, 0))
    return self._lower_border_end