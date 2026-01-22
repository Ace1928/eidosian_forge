from __future__ import absolute_import, division, print_function
import numpy as np
import mido
def write_midi(notes, filename, duration=0.6, velocity=100):
    """
    Write notes to a MIDI file.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    filename : str
        Output MIDI file.
    duration : float, optional
        Note duration if not defined by `notes`.
    velocity : int, optional
        Note velocity if not defined by `notes`.

    Returns
    -------
    numpy array
        Notes (including note length, velocity and channel).

    Notes
    -----
    The note columns format must be (duration, velocity and channel optional):

    'onset time' 'note number' ['duration' ['velocity' ['channel']]]

    """
    from ..utils import expand_notes
    notes = expand_notes(notes, duration, velocity)
    MIDIFile.from_notes(notes).save(filename)