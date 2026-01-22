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
def homogeneity_score(labels_true, labels_pred):
    """Homogeneity metric of a cluster labeling given a ground truth.

    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is not symmetric: switching ``label_true`` with ``label_pred``
    will return the :func:`completeness_score` which will be different in
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
    homogeneity : float
       Score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling.

    See Also
    --------
    completeness_score : Completeness metric of cluster labeling.
    v_measure_score : V-Measure (NMI with arithmetic mean option).

    References
    ----------

    .. [1] `Andrew Rosenberg and Julia Hirschberg, 2007. V-Measure: A
       conditional entropy-based external cluster evaluation measure
       <https://aclweb.org/anthology/D/D07/D07-1043.pdf>`_

    Examples
    --------

    Perfect labelings are homogeneous::

      >>> from sklearn.metrics.cluster import homogeneity_score
      >>> homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Non-perfect labelings that further split classes into more clusters can be
    perfectly homogeneous::

      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 1, 2]))
      1.000000
      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 2, 3]))
      1.000000

    Clusters that include samples from different classes do not make for an
    homogeneous labeling::

      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1]))
      0.0...
      >>> print("%.6f" % homogeneity_score([0, 0, 1, 1], [0, 0, 0, 0]))
      0.0...
    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[0]