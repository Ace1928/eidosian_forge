import numpy as np
import os
import pkg_resources
from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz
def remove_invalid_notes(self):
    """Removes any notes whose end time is before or at their start time.

        """
    notes_to_delete = []
    for note in self.notes:
        if note.end <= note.start:
            notes_to_delete.append(note)
    for note in notes_to_delete:
        self.notes.remove(note)