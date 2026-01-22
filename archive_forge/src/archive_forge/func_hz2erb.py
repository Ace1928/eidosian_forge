from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
def hz2erb(f):
    """
    Convert Hz to ERB.

    Parameters
    ----------
    f : numpy array
        Input frequencies [Hz].

    Returns
    -------
    e : numpy array
        Frequencies in ERB [ERB].

    Notes
    -----
    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return 21.4 * np.log10(1 + 4.37 * np.asarray(f) / 1000.0)