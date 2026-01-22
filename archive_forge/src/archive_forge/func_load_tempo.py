from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
def load_tempo(filename, split_value=1.0, sort=None, norm_strengths=None, max_len=None):
    """
    Load tempo information from the given file.

    Tempo information must have the following format:
    'main tempo' ['secondary tempo' ['relative_strength']]

    Parameters
    ----------
    filename : str or file handle
        File to load the tempo from.
    split_value : float, optional
        Value to distinguish between tempi and strengths.
        `values` > `split_value` are interpreted as tempi [bpm],
        `values` <= `split_value` are interpreted as strengths.
    sort : bool, deprecated
        Sort the tempi by their strength.
    norm_strengths : bool, deprecated
        Normalize the strengths to sum 1.
    max_len : int, deprecated
        Return at most `max_len` tempi.

    Returns
    -------
    tempi : numpy array, shape (num_tempi[, 2])
        Array with tempi. If no strength is parsed, a 1-dimensional array of
        length 'num_tempi' is returned. If strengths are given, a 2D array
        with tempi (first column) and their relative strengths (second column)
        is returned.


    """
    values = np.loadtxt(filename, ndmin=1)
    tempi = values[values > split_value]
    strengths = values[values <= split_value]
    strength_sum = np.sum(strengths)
    if len(tempi) - len(strengths) == 1:
        strengths = np.append(strengths, 1.0 - strength_sum)
        if np.any(strengths < 0):
            raise AssertionError('strengths must be positive')
    if strength_sum == 0:
        strengths = np.ones_like(tempi) / float(len(tempi))
    if norm_strengths is not None:
        import warnings
        warnings.warn('`norm_strengths` is deprecated as of version 0.16 and will be removed in 0.18. Please normalize strengths separately.')
        strengths /= float(strength_sum)
    if len(tempi) != len(strengths):
        raise AssertionError('tempi and strengths must have same length')
    if sort:
        import warnings
        warnings.warn('`sort` is deprecated as of version 0.16 and will be removed in 0.18. Please sort the returned array separately.')
        sort_idx = (-strengths).argsort(kind='mergesort')
        tempi = tempi[sort_idx]
        strengths = strengths[sort_idx]
    if max_len is not None:
        import warnings
        warnings.warn('`max_len` is deprecated as of version 0.16 and will be removed in 0.18. Please truncate the returned array separately.')
    return np.vstack((tempi[:max_len], strengths[:max_len])).T