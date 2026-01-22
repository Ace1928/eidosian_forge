from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
@array
def calc_relative_errors(detections, annotations, matches=None):
    """
    Errors of the detections relative to the closest annotated interval.

    Parameters
    ----------
    detections : list or numpy array
        Detected beats.
    annotations : list or numpy array
        Annotated beats.
    matches : list or numpy array
        Indices of the closest beats.

    Returns
    -------
    numpy array
        Errors relative to the closest annotated beat interval.

    Notes
    -----
    The sequences must be ordered! To speed up the calculation, a list of
    pre-computed indices of the closest matches can be used.

    """
    if len(detections) == 0:
        return np.zeros(0, dtype=np.float)
    if len(annotations) < 2:
        raise BeatIntervalError
    if matches is None:
        matches = find_closest_matches(detections, annotations)
    errors = calc_errors(detections, annotations, matches)
    intervals = find_closest_intervals(detections, annotations, matches)
    return errors / intervals