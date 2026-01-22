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
@validate_params({'reachability': [np.ndarray], 'predecessor': [np.ndarray], 'ordering': [np.ndarray], 'min_samples': [Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both')], 'min_cluster_size': [Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both'), None], 'xi': [Interval(Real, 0, 1, closed='both')], 'predecessor_correction': ['boolean']}, prefer_skip_nested_validation=True)
def cluster_optics_xi(*, reachability, predecessor, ordering, min_samples, min_cluster_size=None, xi=0.05, predecessor_correction=True):
    """Automatically extract clusters according to the Xi-steep method.

    Parameters
    ----------
    reachability : ndarray of shape (n_samples,)
        Reachability distances calculated by OPTICS (`reachability_`).

    predecessor : ndarray of shape (n_samples,)
        Predecessors calculated by OPTICS.

    ordering : ndarray of shape (n_samples,)
        OPTICS ordered point indices (`ordering_`).

    min_samples : int > 1 or float between 0 and 1
        The same as the min_samples given to OPTICS. Up and down steep regions
        can't have more then ``min_samples`` consecutive non-steep points.
        Expressed as an absolute number or a fraction of the number of samples
        (rounded to be at least 2).

    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.

    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.

    predecessor_correction : bool, default=True
        Correct clusters based on the calculated predecessors.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The labels assigned to samples. Points which are not included
        in any cluster are labeled as -1.

    clusters : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to ``(end,
        -start)`` (ascending) so that larger clusters encompassing smaller
        clusters come after such nested smaller clusters. Since ``labels`` does
        not reflect the hierarchy, usually ``len(clusters) >
        np.unique(labels)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import cluster_optics_xi, compute_optics_graph
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
    ...     n_jobs=None
    ... )
    >>> min_samples = 2
    >>> labels, clusters = cluster_optics_xi(
    ...     reachability=reachability,
    ...     predecessor=predecessor,
    ...     ordering=ordering,
    ...     min_samples=min_samples,
    ... )
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    >>> clusters
    array([[0, 2],
           [3, 5],
           [0, 5]])
    """
    n_samples = len(reachability)
    _validate_size(min_samples, n_samples, 'min_samples')
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))
    if min_cluster_size is None:
        min_cluster_size = min_samples
    _validate_size(min_cluster_size, n_samples, 'min_cluster_size')
    if min_cluster_size <= 1:
        min_cluster_size = max(2, int(min_cluster_size * n_samples))
    clusters = _xi_cluster(reachability[ordering], predecessor[ordering], ordering, xi, min_samples, min_cluster_size, predecessor_correction)
    labels = _extract_xi_labels(ordering, clusters)
    return (labels, clusters)