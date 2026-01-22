import warnings
from math import log
from numbers import Real
import numpy as np
from scipy import sparse as sp
from ...utils._param_validation import Interval, StrOptions, validate_params
from ...utils.multiclass import type_of_target
from ...utils.validation import check_array, check_consistent_length
from ._expected_mutual_info_fast import expected_mutual_information
@validate_params({'labels_true': ['array-like'], 'labels_pred': ['array-like']}, prefer_skip_nested_validation=True)
def rand_score(labels_true, labels_pred):
    """Rand index.

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings [1]_ [2]_.

    The raw RI score [3]_ is:

        RI = (number of agreeing pairs) / (number of pairs)

    Read more in the :ref:`User Guide <rand_score>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.

    Returns
    -------
    RI : float
       Similarity score between 0.0 and 1.0, inclusive, 1.0 stands for
       perfect match.

    See Also
    --------
    adjusted_rand_score: Adjusted Rand Score.
    adjusted_mutual_info_score: Adjusted Mutual Information.

    References
    ----------
    .. [1] :doi:`Hubert, L., Arabie, P. "Comparing partitions."
       Journal of Classification 2, 193–218 (1985).
       <10.1007/BF01908075>`.

    .. [2] `Wikipedia: Simple Matching Coefficient
        <https://en.wikipedia.org/wiki/Simple_matching_coefficient>`_

    .. [3] `Wikipedia: Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_

    Examples
    --------
    Perfectly matching labelings have a score of 1 even

      >>> from sklearn.metrics.cluster import rand_score
      >>> rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete but may not always be pure, hence penalized:

      >>> rand_score([0, 0, 1, 2], [0, 0, 1, 1])
      0.83...
    """
    contingency = pair_confusion_matrix(labels_true, labels_pred)
    numerator = contingency.diagonal().sum()
    denominator = contingency.sum()
    if numerator == denominator or denominator == 0:
        return 1.0
    return numerator / denominator