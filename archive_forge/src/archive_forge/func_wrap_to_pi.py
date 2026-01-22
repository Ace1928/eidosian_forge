from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def wrap_to_pi(phase):
    """
    Wrap the phase information to the range -π...π.

    Parameters
    ----------
    phase : numpy array
        Phase of the STFT.

    Returns
    -------
    wrapped_phase : numpy array
        Wrapped phase.

    """
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi