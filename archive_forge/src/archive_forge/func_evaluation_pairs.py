import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def evaluation_pairs(det_chords, ann_chords):
    """
    Match detected with annotated chords and create paired label segments
    for evaluation.

    Parameters
    ----------
    det_chords : numpy structured array
        Chord detections with 'start' and 'end' fields.
    ann_chords : numpy structured array
        Chord annotations with 'start' and 'end' fields.

    Returns
    -------
    annotations : numpy structured array
        Annotated chords of evaluation segments.
    detections : numpy structured array
        Detected chords of evaluation segments.
    durations : numpy array
        Durations of evaluation segments.

    """
    times = np.unique(np.hstack([ann_chords['start'], ann_chords['end'], det_chords['start'], det_chords['end']]))
    durations = times[1:] - times[:-1]
    annotations = ann_chords['chord'][np.searchsorted(ann_chords['start'], times[:-1], side='right') - 1]
    detections = det_chords['chord'][np.searchsorted(det_chords['start'], times[:-1], side='right') - 1]
    return (annotations, detections, durations)