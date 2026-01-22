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
def completeness_score(labels_true, labels_pred):
    """Compute completeness metric of a cluster labeling given a ground truth.

    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is not symmetric: switching ``label_true`` with ``label_pred``
    will return the :func:`homogeneity_score` which will be different in
    general.

    Read more in the :ref:`User Guide <homogeneity_completeness>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    completeness : float
       Score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling.

    See Also
    --------
    homogeneity_score : Homogeneity metric of cluster labeling.
    v_measure_score : V-Measure (NMI with arithmetic mean option).

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    Examples
    --------

    Perfect labelings are complete::

      >>> from sklearn.metrics.cluster import completeness_score
      >>> completeness_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Non-perfect labelings that assign all classes members to the same clusters
    are still complete::

      >>> print(completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))
      1.0
      >>> print(completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))
      0.999...

    If classes members are split across different clusters, the
    assignment cannot be complete::

      >>> print(completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))
      0.0
      >>> print(completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))
      0.0
    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[1]