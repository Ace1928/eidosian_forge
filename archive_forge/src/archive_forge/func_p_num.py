from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def p_num(self, n: int) -> int:
    """Number of time slices for an input signal with `n` samples.

        It is given by `p_num` = `p_max` - `p_min` with `p_min` typically
        being negative.
        A detailed example is provided in the :ref:`tutorial_stft_sliding_win`
        section of the :ref:`user_guide`.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_range: Determine and validate slice index range.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this method belongs to.
        """
    return self.p_max(n) - self.p_min