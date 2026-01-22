from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
@fft_mode.setter
def fft_mode(self, t: FFT_MODE_TYPE):
    """Set mode of FFT.

        Allowed values are 'twosided', 'centered', 'onesided', 'onesided2X'.
        See the property `fft_mode` for more details.
        """
    if t not in (fft_mode_types := get_args(FFT_MODE_TYPE)):
        raise ValueError(f"fft_mode='{t}' not in {fft_mode_types}!")
    if t in {'onesided', 'onesided2X'} and np.iscomplexobj(self.win):
        raise ValueError(f"One-sided spectra, i.e., fft_mode='{t}', " + 'are not allowed for complex-valued windows!')
    if t == 'onesided2X' and self.scaling is None:
        raise ValueError(f"For scaling is None, fft_mode='{t}' is invalid!Do scale_to('psd') or scale_to('magnitude')!")
    self._fft_mode = t