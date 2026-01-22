import cupy
import numpy as np
from cupyx.scipy.signal._signaltools import convolve
def qmf(hk):
    """
    Return high-pass qmf filter from low-pass

    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.

    """
    hk = cupy.asarray(hk)
    return _qmf_kernel(hk, size=len(hk))