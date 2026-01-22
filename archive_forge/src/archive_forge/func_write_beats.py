from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
def write_beats(beats, filename, fmt=None, delimiter='\t', header=None):
    """
    Write the beats to a file.

    Parameters
    ----------
    beats : numpy array
        Beats to be written to file.
    filename : str or file handle
        File to write the beats to.
    fmt : str or sequence of strs, optional
        A single format (e.g. '%.3f'), a sequence of formats (e.g.
        ['%.3f', '%d']), or a multi-format string (e.g. '%.3f %d'), in which
        case `delimiter` is ignored.
    delimiter : str, optional
        String or character separating columns.
    header : str, optional
        String that will be written at the beginning of the file as comment.

    """
    if fmt is None and beats.ndim == 2:
        fmt = ['%.3f', '%d']
    elif fmt is None:
        fmt = '%.3f'
    write_events(beats, filename, fmt, delimiter, header)