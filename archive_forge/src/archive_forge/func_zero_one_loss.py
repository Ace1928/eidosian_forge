import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import LabelBinarizer, LabelEncoder
from ..utils import (
from ..utils._array_api import _union1d, _weighted_sum, get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _nanaverage
from ..utils.multiclass import type_of_target, unique_labels
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _num_samples
@validate_params({'y_true': ['array-like', 'sparse matrix'], 'y_pred': ['array-like', 'sparse matrix'], 'normalize': ['boolean'], 'sample_weight': ['array-like', None]}, prefer_skip_nested_validation=True)
def zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None):
    """Zero-one classification loss.

    If normalize is ``True``, return the fraction of misclassifications
    (float), else it returns the number of misclassifications (int). The best
    performance is 0.

    Read more in the :ref:`User Guide <zero_one_loss>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    normalize : bool, default=True
        If ``False``, return the number of misclassifications.
        Otherwise, return the fraction of misclassifications.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float or int,
        If ``normalize == True``, return the fraction of misclassifications
        (float), else it returns the number of misclassifications (int).

    See Also
    --------
    accuracy_score : Compute the accuracy score. By default, the function will
        return the fraction of correct predictions divided by the total number
        of predictions.
    hamming_loss : Compute the average Hamming loss or Hamming distance between
        two sets of samples.
    jaccard_score : Compute the Jaccard similarity coefficient score.

    Notes
    -----
    In multilabel classification, the zero_one_loss function corresponds to
    the subset zero-one loss: for each sample, the entire set of labels must be
    correctly predicted, otherwise the loss for that sample is equal to one.

    Examples
    --------
    >>> from sklearn.metrics import zero_one_loss
    >>> y_pred = [1, 2, 3, 4]
    >>> y_true = [2, 2, 3, 4]
    >>> zero_one_loss(y_true, y_pred)
    0.25
    >>> zero_one_loss(y_true, y_pred, normalize=False)
    1.0

    In the multilabel case with binary label indicators:

    >>> import numpy as np
    >>> zero_one_loss(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.5
    """
    xp, _ = get_namespace(y_true, y_pred)
    score = accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
    if normalize:
        return 1 - score
    else:
        if sample_weight is not None:
            n_samples = xp.sum(sample_weight)
        else:
            n_samples = _num_samples(y_true)
        return n_samples - score