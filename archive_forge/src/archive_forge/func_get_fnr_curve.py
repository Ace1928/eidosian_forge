from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
def get_fnr_curve(model=None, data=None, curve=None, thread_count=-1, plot=False):
    """
    Build points of FNR curve.

    Parameters
    ----------
    model : catboost.CatBoost
        The trained model.

    data : catboost.Pool or list of catboost.Pool
        A set of samples to build ROC curve with.

    curve : tuple of three arrays (fpr, tpr, thresholds)
        ROC curve points in format of get_roc_curve returned value.
        If set, data parameter must not be set.

    thread_count : int (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    plot : bool, optional (default=False)
        If True, draw curve.

    Returns
    -------
    curve points : tuple of two arrays (thresholds, fnr)
    """
    if curve is not None:
        if data is not None:
            raise CatBoostError('Only one of the parameters data and curve should be set.')
        if not (isinstance(curve, list) or isinstance(curve, tuple)) or len(curve) != 3:
            raise CatBoostError('curve must be list or tuple of three arrays (fpr, tpr, thresholds).')
        tpr, thresholds = (curve[1], curve[2][:])
    else:
        if model is None or data is None:
            raise CatBoostError('model and data parameters should be set when curve parameter is None.')
        _, tpr, thresholds = get_roc_curve(model, data, thread_count)
    fnr = np.array([1 - x for x in tpr])
    if plot:
        with _import_matplotlib() as plt:
            _draw(plt, thresholds, fnr, 'Thresholds', 'False Negative Rate', 'FNR Curve')
    return (thresholds, fnr)