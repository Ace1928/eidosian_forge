from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
def attenuate(signal, attenuation):
    """
    Attenuate the signal.

    Parameters
    ----------
    signal : numpy array
        Signal to be attenuated.
    attenuation :  float
        Attenuation level [dB].

    Returns
    -------
    numpy array
        Attenuated signal (same dtype as `signal`).

    Notes
    -----
    The signal is returned with the same dtype, thus rounding errors may occur
    with integer dtypes.

    """
    if attenuation == 0:
        return signal
    return adjust_gain(signal, -attenuation)