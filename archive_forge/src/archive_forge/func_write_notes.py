from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
def write_notes(notes, filename, fmt=None, delimiter='\t', header=None):
    """
    Write the notes to a file.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, row format 'onset_time' 'note_number' ['duration' ['velocity']].
    filename : str or file handle
        File to write the notes to.
    fmt : str or sequence of strs, optional
        A sequence of formats (e.g. ['%.3f', '%d', '%.3f', '%d']), or a
        multi-format string, e.g. '%.3f %d %.3f %d', in which case `delimiter`
        is ignored.
    delimiter : str, optional
        String or character separating columns.
    header : str, optional
        String that will be written at the beginning of the file as comment.

    Returns
    -------
    numpy array
        Notes.

    """
    if fmt is None:
        fmt = ['%.3f', '%d', '%.3f', '%d']
    if not notes.ndim == 2:
        raise ValueError('unknown format for `notes`')
    fmt = delimiter.join(fmt[:notes.shape[1]])
    write_events(notes, filename, fmt=fmt, delimiter=delimiter, header=header)