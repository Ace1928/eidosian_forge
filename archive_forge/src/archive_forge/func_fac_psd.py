from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@property
def fac_psd(self) -> float:
    """Factor to multiply the STFT values by to scale each frequency slice
        to a power spectral density (PSD).

        It is 1 if attribute ``scaling == 'psd'``.
        The window can be scaled to a psd spectrum by using the method
        `scale_to`.

        See Also
        --------
        fac_magnitude: Scaling factor for to a magnitude spectrum.
        scale_to: Scale window to obtain 'magnitude' or 'psd' scaling.
        scaling: Normalization applied to the window function.
        ShortTimeFFT: Class this property belongs to.
        """
    if self.scaling == 'psd':
        return 1
    if self._fac_psd is None:
        self._fac_psd = 1 / np.sqrt(sum(self.win.real ** 2 + self.win.imag ** 2) / self.T)
    return self._fac_psd