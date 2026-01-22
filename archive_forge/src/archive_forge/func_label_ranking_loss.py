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
@validate_params({'y_true': ['array-like', 'sparse matrix'], 'y_score': ['array-like'], 'sample_weight': ['array-like', None]}, prefer_skip_nested_validation=True)
def label_ranking_loss(y_true, y_score, *, sample_weight=None):
    """Compute Ranking loss measure.

    Compute the average number of label pairs that are incorrectly ordered
    given y_score weighted by the size of the label set and the number of
    labels not in the label set.

    This is similar to the error set size, but weighted by the number of
    relevant and irrelevant labels. The best performance is achieved with
    a ranking loss of zero.

    Read more in the :ref:`User Guide <label_ranking_loss>`.

    .. versionadded:: 0.17
       A function *label_ranking_loss*

    Parameters
    ----------
    y_true : {array-like, sparse matrix} of shape (n_samples, n_labels)
        True binary labels in binary indicator format.

    y_score : array-like of shape (n_samples, n_labels)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        Average number of label pairs that are incorrectly ordered given
        y_score weighted by the size of the label set and the number of labels not
        in the label set.

    References
    ----------
    .. [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010).
           Mining multi-label data. In Data mining and knowledge discovery
           handbook (pp. 667-685). Springer US.

    Examples
    --------
    >>> from sklearn.metrics import label_ranking_loss
    >>> y_true = [[1, 0, 0], [0, 0, 1]]
    >>> y_score = [[0.75, 0.5, 1], [1, 0.2, 0.1]]
    >>> label_ranking_loss(y_true, y_score)
    0.75...
    """
    y_true = check_array(y_true, ensure_2d=False, accept_sparse='csr')
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)
    y_type = type_of_target(y_true, input_name='y_true')
    if y_type not in ('multilabel-indicator',):
        raise ValueError('{0} format is not supported'.format(y_type))
    if y_true.shape != y_score.shape:
        raise ValueError('y_true and y_score have different shape')
    n_samples, n_labels = y_true.shape
    y_true = csr_matrix(y_true)
    loss = np.zeros(n_samples)
    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
        unique_scores, unique_inverse = np.unique(y_score[i], return_inverse=True)
        true_at_reversed_rank = np.bincount(unique_inverse[y_true.indices[start:stop]], minlength=len(unique_scores))
        all_at_reversed_rank = np.bincount(unique_inverse, minlength=len(unique_scores))
        false_at_reversed_rank = all_at_reversed_rank - true_at_reversed_rank
        loss[i] = np.dot(true_at_reversed_rank.cumsum(), false_at_reversed_rank)
    n_positives = count_nonzero(y_true, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        loss /= (n_labels - n_positives) * n_positives
    loss[np.logical_or(n_positives == 0, n_positives == n_labels)] = 0.0
    return np.average(loss, weights=sample_weight)