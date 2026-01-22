import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def chords(labels):
    """
    Transform a list of chord labels into an array of internal numeric
    representations.

    Parameters
    ----------
    labels : list
        List of chord labels (str).

    Returns
    -------
    chords : numpy.array
        Structured array with columns 'root', 'bass', and 'intervals',
        containing a numeric representation of chords (`CHORD_DTYPE`).

    """
    crds = np.zeros(len(labels), dtype=CHORD_DTYPE)
    cache = {}
    for i, lbl in enumerate(labels):
        cv = cache.get(lbl, None)
        if cv is None:
            cv = chord(lbl)
            cache[lbl] = cv
        crds[i] = cv
    return crds