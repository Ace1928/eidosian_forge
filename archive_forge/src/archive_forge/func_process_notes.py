from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
def process_notes(data, output=None):
    """
    This is a simple processing function. It either loads the notes from a MIDI
    file and or writes the notes to a file.

    The behaviour depends on the presence of the `output` argument, if 'None'
    is given, the notes are read, otherwise the notes are written to file.

    Parameters
    ----------
    data : str or numpy array
        MIDI file to be loaded (if `output` is 'None') / notes to be written.
    output : str, optional
        Output file name. If set, the notes given by `data` are written.

    Returns
    -------
    notes : numpy array
        Notes read/written.

    """
    if output is None:
        return MIDIFile.from_file(data).notes()
    MIDIFile.from_notes(data).write(output)
    return data