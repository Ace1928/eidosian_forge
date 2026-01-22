from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
def load_segments(filename):
    """
    Load labelled segments from file, one segment per line. Each segment is of
    form <start> <end> <label>, where <start> and <end> are floating point
    numbers, and <label> is a string.

    Parameters
    ----------
    filename : str or file handle
        File to read the labelled segments from.

    Returns
    -------
    segments : numpy structured array
        Structured array with columns 'start', 'end', and 'label',
        containing the beginning, end, and label of segments.

    """
    start, end, label = ([], [], [])
    with open_file(filename) as f:
        for line in f:
            s, e, l = line.split()
            start.append(float(s))
            end.append(float(e))
            label.append(l)
    segments = np.zeros(len(start), dtype=SEGMENT_DTYPE)
    segments['start'] = start
    segments['end'] = end
    segments['label'] = label
    return segments