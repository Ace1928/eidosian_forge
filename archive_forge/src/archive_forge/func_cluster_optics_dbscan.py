import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import SparseEfficiencyWarning, issparse
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..exceptions import DataConversionWarning
from ..metrics import pairwise_distances
from ..metrics.pairwise import _VALID_METRICS, PAIRWISE_BOOLEAN_FUNCTIONS
from ..neighbors import NearestNeighbors
from ..utils import gen_batches, get_chunk_n_rows
from ..utils._param_validation import (
from ..utils.validation import check_memory
@validate_params({'reachability': [np.ndarray], 'core_distances': [np.ndarray], 'ordering': [np.ndarray], 'eps': [Interval(Real, 0, None, closed='both')]}, prefer_skip_nested_validation=True)
def cluster_optics_dbscan(*, reachability, core_distances, ordering, eps):
    """Perform DBSCAN extraction for an arbitrary epsilon.

    Extracting the clusters runs in linear time. Note that this results in
    ``labels_`` which are close to a :class:`~sklearn.cluster.DBSCAN` with
    similar settings and ``eps``, only if ``eps`` is close to ``max_eps``.

    Parameters
    ----------
    reachability : ndarray of shape (n_samples,)
        Reachability distances calculated by OPTICS (``reachability_``).

    core_distances : ndarray of shape (n_samples,)
        Distances at which points become core (``core_distances_``).

    ordering : ndarray of shape (n_samples,)
        OPTICS ordered point indices (``ordering_``).

    eps : float
        DBSCAN ``eps`` parameter. Must be set to < ``max_eps``. Results
        will be close to DBSCAN algorithm if ``eps`` and ``max_eps`` are close
        to one another.

    Returns
    -------
    labels_ : array of shape (n_samples,)
        The estimated labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import cluster_optics_dbscan, compute_optics_graph
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> ordering, core_distances, reachability, predecessor = compute_optics_graph(
    ...     X,
    ...     min_samples=2,
    ...     max_eps=np.inf,
    ...     metric="minkowski",
    ...     p=2,
    ...     metric_params=None,
    ...     algorithm="auto",
    ...     leaf_size=30,
    ...     n_jobs=None,
    ... )
    >>> eps = 4.5
    >>> labels = cluster_optics_dbscan(
    ...     reachability=reachability,
    ...     core_distances=core_distances,
    ...     ordering=ordering,
    ...     eps=eps,
    ... )
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    """
    n_samples = len(core_distances)
    labels = np.zeros(n_samples, dtype=int)
    far_reach = reachability > eps
    near_core = core_distances <= eps
    labels[ordering] = np.cumsum(far_reach[ordering] & near_core[ordering]) - 1
    labels[far_reach & ~near_core] = -1
    return labels