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
@validate_params({'X': [np.ndarray, 'sparse matrix'], 'min_samples': [Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both')], 'max_eps': [Interval(Real, 0, None, closed='both')], 'metric': [StrOptions(set(_VALID_METRICS) | {'precomputed'}), callable], 'p': [Interval(Real, 0, None, closed='right'), None], 'metric_params': [dict, None], 'algorithm': [StrOptions({'auto', 'brute', 'ball_tree', 'kd_tree'})], 'leaf_size': [Interval(Integral, 1, None, closed='left')], 'n_jobs': [Integral, None]}, prefer_skip_nested_validation=False)
def compute_optics_graph(X, *, min_samples, max_eps, metric, p, metric_params, algorithm, leaf_size, n_jobs):
    """Compute the OPTICS reachability graph.

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features), or             (n_samples, n_samples) if metric='precomputed'
        A feature array, or array of distances between samples if
        metric='precomputed'.

    min_samples : int > 1 or float between 0 and 1
        The number of samples in a neighborhood for a point to be considered
        as a core point. Expressed as an absolute number or a fraction of the
        number of samples (rounded to be at least 2).

    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string. If metric is
        "precomputed", X is assumed to be a distance matrix and must be square.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

        .. note::
           `'kulsinski'` is deprecated from SciPy 1.9 and will be removed in SciPy 1.11.

    p : float, default=2
        Parameter for the Minkowski metric from
        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`~sklearn.neighbors.BallTree`.
        - 'kd_tree' will use :class:`~sklearn.neighbors.KDTree`.
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to `fit` method. (default)

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
        construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    ordering_ : array of shape (n_samples,)
        The cluster ordered list of sample indices.

    core_distances_ : array of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    reachability_ : array of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    predecessor_ : array of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.

    References
    ----------
    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
       and Jörg Sander. "OPTICS: ordering points to identify the clustering
       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import compute_optics_graph
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
    >>> ordering
    array([0, 1, 2, 5, 3, 4])
    >>> core_distances
    array([3.16..., 1.41..., 1.41..., 1.        , 1.        ,
           4.12...])
    >>> reachability
    array([       inf, 3.16..., 1.41..., 4.12..., 1.        ,
           5.        ])
    >>> predecessor
    array([-1,  0,  1,  5,  3,  2])
    """
    n_samples = X.shape[0]
    _validate_size(min_samples, n_samples, 'min_samples')
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))
    reachability_ = np.empty(n_samples)
    reachability_.fill(np.inf)
    predecessor_ = np.empty(n_samples, dtype=int)
    predecessor_.fill(-1)
    nbrs = NearestNeighbors(n_neighbors=min_samples, algorithm=algorithm, leaf_size=leaf_size, metric=metric, metric_params=metric_params, p=p, n_jobs=n_jobs)
    nbrs.fit(X)
    core_distances_ = _compute_core_distances_(X=X, neighbors=nbrs, min_samples=min_samples, working_memory=None)
    core_distances_[core_distances_ > max_eps] = np.inf
    np.around(core_distances_, decimals=np.finfo(core_distances_.dtype).precision, out=core_distances_)
    processed = np.zeros(X.shape[0], dtype=bool)
    ordering = np.zeros(X.shape[0], dtype=int)
    for ordering_idx in range(X.shape[0]):
        index = np.where(processed == 0)[0]
        point = index[np.argmin(reachability_[index])]
        processed[point] = True
        ordering[ordering_idx] = point
        if core_distances_[point] != np.inf:
            _set_reach_dist(core_distances_=core_distances_, reachability_=reachability_, predecessor_=predecessor_, point_index=point, processed=processed, X=X, nbrs=nbrs, metric=metric, metric_params=metric_params, p=p, max_eps=max_eps)
    if np.all(np.isinf(reachability_)):
        warnings.warn('All reachability values are inf. Set a larger max_eps or all data will be considered outliers.', UserWarning)
    return (ordering, core_distances_, reachability_, predecessor_)