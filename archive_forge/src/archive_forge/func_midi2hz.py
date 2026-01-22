from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
def midi2hz(m, fref=A4):
    """
    Convert MIDI notes to corresponding frequencies.

    Parameters
    ----------
    m : numpy array
        Input MIDI notes.
    fref : float, optional
        Tuning frequency of A4 [Hz].

    Returns
    -------
    f : numpy array
        Corresponding frequencies [Hz].

    """
    return 2.0 ** ((np.asarray(m, dtype=np.float) - 69.0) / 12.0) * fref