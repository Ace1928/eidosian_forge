from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import EvaluationMixin, MeanEvaluation, evaluation_io
from ..io import load_tempo
def tempo_evaluation(detections, annotations, tolerance=TOLERANCE):
    """
    Calculate the tempo P-Score, at least one and all tempi correct.

    Parameters
    ----------
    detections : list of tuples or numpy array
        Detected tempi (rows, first column) and their relative strengths
        (second column).
    annotations : list or numpy array
        Annotated tempi (rows, first column) and their relative strengths
        (second column).
    tolerance : float, optional
        Evaluation tolerance (max. allowed deviation).

    Returns
    -------
    pscore : float
        P-Score.
    at_least_one : bool
        At least one tempo correctly identified.
    all : bool
        All tempi correctly identified.

    Notes
    -----
    All given detections are evaluated against all annotations according to the
    relative strengths given. If no strengths are given, evenly distributed
    strengths are assumed. If the strengths do not sum to 1, they will be
    normalized.

    References
    ----------
    .. [1] M. McKinney, D. Moelants, M. Davies and A. Klapuri,
           "Evaluation of audio beat tracking and music tempo extraction
           algorithms",
           Journal of New Music Research, vol. 36, no. 1, 2007.

    """
    if len(detections) == 0 and len(annotations) == 0:
        return (1.0, True, True)
    if len(detections) == 0 or len(annotations) == 0:
        return (0.0, False, False)
    if float(tolerance) <= 0:
        raise ValueError('tolerance must be greater than 0')
    detections = np.array(detections, dtype=np.float, ndmin=1)
    annotations = np.array(annotations, dtype=np.float, ndmin=1)
    if detections.ndim == 2:
        detections = detections[:, 0]
    strengths = []
    if annotations.ndim == 2:
        strengths = annotations[:, 1]
        annotations = annotations[:, 0]
    strengths_sum = np.sum(strengths)
    if strengths_sum == 0:
        warnings.warn('no annotated tempo strengths given, assuming a uniform distribution')
        strengths = np.ones_like(annotations) / float(len(annotations))
    elif strengths_sum != 1:
        warnings.warn('annotated tempo strengths do not sum to 1, normalizing')
        strengths /= float(strengths_sum)
    errors = np.abs(1 - detections[:, np.newaxis] / annotations)
    correct = np.asarray(np.sum(errors <= tolerance, axis=0), np.bool)
    pscore = np.sum(strengths[correct])
    return (pscore, correct.any(), correct.all())