import math
import warnings
import numpy as np
import dataclasses
from typing import Optional, Callable
from functools import partial
from scipy._lib._util import _asarray_validated
from . import _distance_wrap
from . import _hausdorff
from ..linalg import norm
from ..special import rel_entr
from . import _distance_pybind
def directed_hausdorff(u, v, seed=0):
    """
    Compute the directed Hausdorff distance between two 2-D arrays.

    Distances between pairs are calculated using a Euclidean metric.

    Parameters
    ----------
    u : (M,N) array_like
        Input array with M points in N dimensions.
    v : (O,N) array_like
        Input array with O points in N dimensions.
    seed : int or None
        Local `numpy.random.RandomState` seed. Default is 0, a random
        shuffling of u and v that guarantees reproducibility.

    Returns
    -------
    d : double
        The directed Hausdorff distance between arrays `u` and `v`,

    index_1 : int
        index of point contributing to Hausdorff pair in `u`

    index_2 : int
        index of point contributing to Hausdorff pair in `v`

    Raises
    ------
    ValueError
        An exception is thrown if `u` and `v` do not have
        the same number of columns.

    See Also
    --------
    scipy.spatial.procrustes : Another similarity test for two data sets

    Notes
    -----
    Uses the early break technique and the random sampling approach
    described by [1]_. Although worst-case performance is ``O(m * o)``
    (as with the brute force algorithm), this is unlikely in practice
    as the input data would have to require the algorithm to explore
    every single point interaction, and after the algorithm shuffles
    the input points at that. The best case performance is O(m), which
    is satisfied by selecting an inner loop distance that is less than
    cmax and leads to an early break as often as possible. The authors
    have formally shown that the average runtime is closer to O(m).

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] A. A. Taha and A. Hanbury, "An efficient algorithm for
           calculating the exact Hausdorff distance." IEEE Transactions On
           Pattern Analysis And Machine Intelligence, vol. 37 pp. 2153-63,
           2015.

    Examples
    --------
    Find the directed Hausdorff distance between two 2-D arrays of
    coordinates:

    >>> from scipy.spatial.distance import directed_hausdorff
    >>> import numpy as np
    >>> u = np.array([(1.0, 0.0),
    ...               (0.0, 1.0),
    ...               (-1.0, 0.0),
    ...               (0.0, -1.0)])
    >>> v = np.array([(2.0, 0.0),
    ...               (0.0, 2.0),
    ...               (-2.0, 0.0),
    ...               (0.0, -4.0)])

    >>> directed_hausdorff(u, v)[0]
    2.23606797749979
    >>> directed_hausdorff(v, u)[0]
    3.0

    Find the general (symmetric) Hausdorff distance between two 2-D
    arrays of coordinates:

    >>> max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    3.0

    Find the indices of the points that generate the Hausdorff distance
    (the Hausdorff pair):

    >>> directed_hausdorff(v, u)[1:]
    (3, 3)

    """
    u = np.asarray(u, dtype=np.float64, order='c')
    v = np.asarray(v, dtype=np.float64, order='c')
    if u.shape[1] != v.shape[1]:
        raise ValueError('u and v need to have the same number of columns')
    result = _hausdorff.directed_hausdorff(u, v, seed)
    return result