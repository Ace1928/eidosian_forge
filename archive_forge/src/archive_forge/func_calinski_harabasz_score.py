import functools
from numbers import Integral
import numpy as np
from scipy.sparse import issparse
from ...preprocessing import LabelEncoder
from ...utils import _safe_indexing, check_random_state, check_X_y
from ...utils._param_validation import (
from ..pairwise import _VALID_METRICS, pairwise_distances, pairwise_distances_chunked
@validate_params({'X': ['array-like'], 'labels': ['array-like']}, prefer_skip_nested_validation=True)
def calinski_harabasz_score(X, labels):
    """Compute the Calinski and Harabasz score.

    It is also known as the Variance Ratio Criterion.

    The score is defined as ratio of the sum of between-cluster dispersion and
    of within-cluster dispersion.

    Read more in the :ref:`User Guide <calinski_harabasz_index>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.

    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.metrics import calinski_harabasz_score
    >>> X, _ = make_blobs(random_state=0)
    >>> kmeans = KMeans(n_clusters=3, random_state=0,).fit(X)
    >>> calinski_harabasz_score(X, kmeans.labels_)
    114.8...
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)
    extra_disp, intra_disp = (0.0, 0.0)
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)
    return 1.0 if intra_disp == 0.0 else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))