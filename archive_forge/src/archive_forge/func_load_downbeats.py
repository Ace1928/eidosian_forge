from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
def load_downbeats(filename):
    """
    Load the downbeats from the given file.

    Parameters
    ----------
    filename : str or file handle
        File to load the downbeats from.

    Returns
    -------
    numpy array
        Downbeats.

    """
    return load_beats(filename, downbeats=True)