from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
def find_closest_intervals(detections, annotations, matches=None):
    """
    Find the closest annotated interval to each beat detection.

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
        Closest annotated beat intervals.

    Notes
    -----
    The sequences must be ordered. To speed up the calculation, a list of
    pre-computed indices of the closest matches can be used.

    The function does NOT test if each detection has a surrounding interval,
    it always returns the closest interval.

    """
    if len(detections) == 0:
        return np.zeros(0, dtype=np.float)
    if len(annotations) < 2:
        raise BeatIntervalError
    closest_interval = np.ones_like(detections)
    intervals = np.zeros(len(annotations) + 1)
    intervals[1:-1] = np.diff(annotations)
    intervals[0] = intervals[1]
    intervals[-1] = intervals[-2]
    if matches is None:
        matches = find_closest_matches(detections, annotations)
    errors = calc_errors(detections, annotations, matches)
    closest_interval[errors > 0] = intervals[matches[errors > 0] + 1]
    closest_interval[errors <= 0] = intervals[matches[errors <= 0]]
    return closest_interval