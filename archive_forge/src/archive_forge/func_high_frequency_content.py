from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def high_frequency_content(spectrogram):
    """
    High Frequency Content.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram instance.

    Returns
    -------
    high_frequency_content : numpy array
        High frequency content onset detection function.

    References
    ----------
    .. [1] Paul Masri,
           "Computer Modeling of Sound for Transformation and Synthesis of
           Musical Signals",
           PhD thesis, University of Bristol, 1996.

    """
    hfc = spectrogram * np.arange(spectrogram.num_bins)
    return np.asarray(np.mean(hfc, axis=1))