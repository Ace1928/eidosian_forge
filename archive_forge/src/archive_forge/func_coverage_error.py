import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import label_binarize
from ..utils import (
from ..utils._encode import _encode, _unique
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import stable_cumsum
from ..utils.fixes import trapezoid
from ..utils.multiclass import type_of_target
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _check_sample_weight
from ._base import _average_binary_score, _average_multiclass_ovo_score
@validate_params({'y_true': ['array-like'], 'y_score': ['array-like'], 'sample_weight': ['array-like', None]}, prefer_skip_nested_validation=True)
def coverage_error(y_true, y_score, *, sample_weight=None):
    """Coverage error measure.

    Compute how far we need to go through the ranked scores to cover all
    true labels. The best value is equal to the average number
    of labels in ``y_true`` per sample.

    Ties in ``y_scores`` are broken by giving maximal rank that would have
    been assigned to all tied values.

    Note: Our implementation's score is 1 greater than the one given in
    Tsoumakas et al., 2010. This extends it to handle the degenerate case
    in which an instance has 0 true labels.

    Read more in the :ref:`User Guide <coverage_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        True binary labels in binary indicator format.

    y_score : array-like of shape (n_samples, n_labels)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    coverage_error : float
        The coverage error.

    References
    ----------
    .. [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010).
           Mining multi-label data. In Data mining and knowledge discovery
           handbook (pp. 667-685). Springer US.

    Examples
    --------
    >>> from sklearn.metrics import coverage_error
    >>> y_true = [[1, 0, 0], [0, 1, 1]]
    >>> y_score = [[1, 0, 0], [0, 1, 1]]
    >>> coverage_error(y_true, y_score)
    1.5
    """
    y_true = check_array(y_true, ensure_2d=True)
    y_score = check_array(y_score, ensure_2d=True)
    check_consistent_length(y_true, y_score, sample_weight)
    y_type = type_of_target(y_true, input_name='y_true')
    if y_type != 'multilabel-indicator':
        raise ValueError('{0} format is not supported'.format(y_type))
    if y_true.shape != y_score.shape:
        raise ValueError('y_true and y_score have different shape')
    y_score_mask = np.ma.masked_array(y_score, mask=np.logical_not(y_true))
    y_min_relevant = y_score_mask.min(axis=1).reshape((-1, 1))
    coverage = (y_score >= y_min_relevant).sum(axis=1)
    coverage = coverage.filled(0)
    return np.average(coverage, weights=sample_weight)