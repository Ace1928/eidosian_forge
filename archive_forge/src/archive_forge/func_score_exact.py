import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def score_exact(det_chords, ann_chords):
    """
    Score similarity of chords. Returns 1 if all chord information (root,
    bass, and intervals) match exactly.

    Parameters
    ----------
    det_chords : numpy structured array
        Detected chords.
    ann_chords : numpy structured array
        Annotated chords.

    Returns
    -------
    scores : numpy array
        Similarity score for each chord.

    """
    return ((ann_chords['root'] == det_chords['root']) & (ann_chords['bass'] == det_chords['bass']) & (ann_chords['intervals'] == det_chords['intervals']).all(axis=1)).astype(np.float)