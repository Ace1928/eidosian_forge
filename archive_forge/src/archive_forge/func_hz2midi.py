from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
def hz2midi(f, fref=A4):
    """
    Convert frequencies to the corresponding MIDI notes.

    Parameters
    ----------
    f : numpy array
        Input frequencies [Hz].
    fref : float, optional
        Tuning frequency of A4 [Hz].

    Returns
    -------
    m : numpy array
        MIDI notes

    Notes
    -----
    For details see: at http://www.phys.unsw.edu.au/jw/notes.html
    This function does not necessarily return a valid MIDI Note, you may need
    to round it to the nearest integer.

    """
    return 12.0 * np.log2(np.asarray(f, dtype=np.float) / fref) + 69.0