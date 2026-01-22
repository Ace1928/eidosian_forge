import warnings
from collections import defaultdict
from numbers import Integral, Real
import numpy as np
from .._config import config_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..metrics.pairwise import pairwise_distances_argmin
from ..neighbors import NearestNeighbors
from ..utils import check_array, check_random_state, gen_batches
from ..utils._param_validation import Interval, validate_params
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
@validate_params({'X': ['array-like']}, prefer_skip_nested_validation=False)
def mean_shift(X, *, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, max_iter=300, n_jobs=None):
    """Perform mean shift clustering of data using a flat kernel.

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Input data.

    bandwidth : float, default=None
        Kernel bandwidth. If not None, must be in the range [0, +inf).

        If None, the bandwidth is determined using a heuristic based on
        the median of all pairwise distances. This will take quadratic time in
        the number of samples. The sklearn.cluster.estimate_bandwidth function
        can be used to do this more efficiently.

    seeds : array-like of shape (n_seeds, n_features) or None
        Point used as initial kernel locations. If None and bin_seeding=False,
        each data point is used as a seed. If None and bin_seeding=True,
        see bin_seeding.

    bin_seeding : bool, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        Ignored if seeds argument is not None.

    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.

    cluster_all : bool, default=True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    max_iter : int, default=300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.

    n_jobs : int, default=None
        The number of jobs to use for the computation. The following tasks benefit
        from the parallelization:

        - The search of nearest neighbors for bandwidth estimation and label
          assignments. See the details in the docstring of the
          ``NearestNeighbors`` class.
        - Hill-climbing optimization for all seeds.

        See :term:`Glossary <n_jobs>` for more details.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.17
           Parallel Execution using *n_jobs*.

    Returns
    -------

    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_mean_shift.py
    <sphx_glr_auto_examples_cluster_plot_mean_shift.py>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import mean_shift
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> cluster_centers, labels = mean_shift(X, bandwidth=2)
    >>> cluster_centers
    array([[3.33..., 6.     ],
           [1.33..., 0.66...]])
    >>> labels
    array([1, 1, 1, 0, 0, 0])
    """
    model = MeanShift(bandwidth=bandwidth, seeds=seeds, min_bin_freq=min_bin_freq, bin_seeding=bin_seeding, cluster_all=cluster_all, n_jobs=n_jobs, max_iter=max_iter).fit(X)
    return (model.cluster_centers_, model.labels_)