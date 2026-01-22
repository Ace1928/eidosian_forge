from __future__ import absolute_import, division, print_function
import numpy as np
from . import Evaluation, MeanEvaluation, SumEvaluation, evaluation_io
from ..io import load_onsets
from ..utils import combine_events
def onset_evaluation(detections, annotations, window=WINDOW):
    """
    Determine the true/false positive/negative detections.

    Parameters
    ----------
    detections : numpy array
        Detected notes.
    annotations : numpy array
        Annotated ground truth notes.
    window : float, optional
        Evaluation window [seconds].

    Returns
    -------
    tp : numpy array, shape (num_tp,)
        True positive detections.
    fp : numpy array, shape (num_fp,)
        False positive detections.
    tn : numpy array, shape (0,)
        True negative detections (empty, see notes).
    fn : numpy array, shape (num_fn,)
        False negative detections.
    errors : numpy array, shape (num_tp,)
        Errors of the true positive detections wrt. the annotations.

    Notes
    -----
    The returned true negative array is empty, because we are not interested
    in this class, since it is magnitudes bigger than true positives array.

    """
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    if detections.ndim > 1 or annotations.ndim > 1:
        raise NotImplementedError('please implement multi-dim support')
    tp = np.zeros(0)
    fp = np.zeros(0)
    tn = np.zeros(0)
    fn = np.zeros(0)
    errors = np.zeros(0)
    if len(detections) == 0 and len(annotations) == 0:
        return (tp, fp, tn, fn, errors)
    elif len(annotations) == 0:
        return (tp, detections, tn, fn, errors)
    elif len(detections) == 0:
        return (tp, fp, tn, annotations, errors)
    if float(window) <= 0:
        raise ValueError('window must be greater than 0')
    det = np.sort(detections)
    ann = np.sort(annotations)
    det_length = len(detections)
    ann_length = len(annotations)
    det_index = 0
    ann_index = 0
    while det_index < det_length and ann_index < ann_length:
        d = det[det_index]
        a = ann[ann_index]
        if abs(d - a) <= window:
            tp = np.append(tp, d)
            errors = np.append(errors, d - a)
            det_index += 1
            ann_index += 1
        elif d < a:
            fp = np.append(fp, d)
            det_index += 1
        elif d > a:
            fn = np.append(fn, a)
            ann_index += 1
        else:
            raise AssertionError('can not match % with %', d, a)
    fp = np.append(fp, det[det_index:])
    fn = np.append(fn, ann[ann_index:])
    if len(tp) + len(fp) != len(detections):
        raise AssertionError('bad TP / FP calculation')
    if len(tp) + len(fn) != len(annotations):
        raise AssertionError('bad FN calculation')
    if len(tp) != len(errors):
        raise AssertionError('bad errors calculation')
    return (np.array(tp), np.array(fp), tn, np.array(fn), np.array(errors))