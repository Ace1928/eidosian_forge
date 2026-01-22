from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def k_min(self) -> int:
    """The smallest possible signal index of the STFT.

        `k_min` is the index of the left-most non-zero value of the lowest
        slice `p_min`. Since the zeroth slice is centered over the zeroth
        sample of the input signal, `k_min` is never positive.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this property belongs to.
        """
    return self._pre_padding()[0]