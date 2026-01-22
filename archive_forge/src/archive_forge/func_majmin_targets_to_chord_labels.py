from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor
def majmin_targets_to_chord_labels(targets, fps):
    """
    Converts a series of major/minor chord targets to human readable chord
    labels. Targets are assumed to be spaced equidistant in time as defined
    by the `fps` parameter (each target represents one 'frame').

    Ids 0-11 encode major chords starting with root 'A', 12-23 minor chords.
    Id 24 represents 'N', the no-chord class.

    Parameters
    ----------
    targets : iterable
        Iterable containing chord class ids.
    fps : float
        Frames per second. Consecutive class

    Returns
    -------
    chord labels : list
        List of tuples of the form (start time, end time, chord label)

    """
    pitch_class_to_label = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    def pred_to_cl(pred):
        """
        Map a class id to a chord label.
        0..11 major chords, 12..23 minor chords, 24 no chord
        """
        if pred == 24:
            return 'N'
        return '{}:{}'.format(pitch_class_to_label[pred % 12], 'maj' if pred < 12 else 'min')
    spf = 1.0 / fps
    labels = [(i * spf, pred_to_cl(p)) for i, p in enumerate(targets)]
    prev_label = (None, None)
    uniq_labels = []
    for label in labels:
        if label[1] != prev_label[1]:
            uniq_labels.append(label)
            prev_label = label
    start_times, chord_labels = zip(*uniq_labels)
    end_times = start_times[1:] + (labels[-1][0] + spf,)
    return np.array(list(zip(start_times, end_times, chord_labels)), dtype=SEGMENT_DTYPE)