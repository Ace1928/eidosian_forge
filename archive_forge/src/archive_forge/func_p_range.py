from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def p_range(self, n: int, p0: int | None=None, p1: int | None=None) -> tuple[int, int]:
    """Determine and validate slice index range.

        Parameters
        ----------
        n : int
            Number of samples of input signal, assuming t[0] = 0.
        p0 : int | None
            First slice index. If 0 then the first slice is centered at t = 0.
            If ``None`` then `p_min` is used. Note that p0 may be < 0 if
            slices are left of t = 0.
        p1 : int | None
            End of interval (last value is p1-1).
            If ``None`` then `p_max(n)` is used.


        Returns
        -------
        p0_ : int
            The fist slice index
        p1_ : int
            End of interval (last value is p1-1).

        Notes
        -----
        A ``ValueError`` is raised if ``p_min <= p0 < p1 <= p_max(n)`` does not
        hold.

        See Also
        --------
        k_min: The smallest possible signal index.
        k_max: First sample index after signal end not touched by a time slice.
        lower_border_end: Where pre-padding effects end.
        p_min: The smallest possible slice index.
        p_max: Index of first non-overlapping upper time slice.
        p_num: Number of time slices, i.e., `p_max` - `p_min`.
        upper_border_begin: Where post-padding effects start.
        ShortTimeFFT: Class this property belongs to.
        """
    p_max = self.p_max(n)
    p0_ = self.p_min if p0 is None else p0
    p1_ = p_max if p1 is None else p1
    if not self.p_min <= p0_ < p1_ <= p_max:
        raise ValueError(f'Invalid Parameter p0={p0!r}, p1={p1!r}, i.e., ' + f'self.p_min={self.p_min!r} <= p0 < p1 <= p_max={p_max!r} ' + f'does not hold for signal length n={n!r}!')
    return (p0_, p1_)