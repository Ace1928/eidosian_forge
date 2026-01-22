from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def complex_domain(spectrogram):
    """
    Complex Domain.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.

    Returns
    -------
    complex_domain : numpy array
        Complex domain onset detection function.

    References
    ----------
    .. [1] Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler,
           "On the use of phase and energy for musical onset detection in the
           complex domain",
           IEEE Signal Processing Letters, Volume 11, Number 6, 2004.

    """
    return np.asarray(np.sum(np.abs(_complex_domain(spectrogram)), axis=1))