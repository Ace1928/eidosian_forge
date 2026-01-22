from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
def erb2hz(e):
    """
    Convert ERB scaled frequencies to Hz.

    Parameters
    ----------
    e : numpy array
        Input frequencies [ERB].

    Returns
    -------
    f : numpy array
        Frequencies in Hz [Hz].

    Notes
    -----
    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return (10.0 ** (np.asarray(e) / 21.4) - 1.0) * 1000.0 / 4.37