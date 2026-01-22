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
def squareform(X, force='no', checks=True):
    """
    Convert a vector-form distance vector to a square-form distance
    matrix, and vice-versa.

    Parameters
    ----------
    X : array_like
        Either a condensed or redundant distance matrix.
    force : str, optional
        As with MATLAB(TM), if force is equal to ``'tovector'`` or
        ``'tomatrix'``, the input will be treated as a distance matrix or
        distance vector respectively.
    checks : bool, optional
        If set to False, no checks will be made for matrix
        symmetry nor zero diagonals. This is useful if it is known that
        ``X - X.T1`` is small and ``diag(X)`` is close to zero.
        These values are ignored any way so they do not disrupt the
        squareform transformation.

    Returns
    -------
    Y : ndarray
        If a condensed distance matrix is passed, a redundant one is
        returned, or if a redundant one is passed, a condensed distance
        matrix is returned.

    Notes
    -----
    1. ``v = squareform(X)``

       Given a square n-by-n symmetric distance matrix ``X``,
       ``v = squareform(X)`` returns a ``n * (n-1) / 2``
       (i.e. binomial coefficient n choose 2) sized vector `v`
       where :math:`v[{n \\choose 2} - {n-i \\choose 2} + (j-i-1)]`
       is the distance between distinct points ``i`` and ``j``.
       If ``X`` is non-square or asymmetric, an error is raised.

    2. ``X = squareform(v)``

       Given a ``n * (n-1) / 2`` sized vector ``v``
       for some integer ``n >= 1`` encoding distances as described,
       ``X = squareform(v)`` returns a n-by-n distance matrix ``X``.
       The ``X[i, j]`` and ``X[j, i]`` values are set to
       :math:`v[{n \\choose 2} - {n-i \\choose 2} + (j-i-1)]`
       and all diagonal elements are zero.

    In SciPy 0.19.0, ``squareform`` stopped casting all input types to
    float64, and started returning arrays of the same dtype as the input.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.distance import pdist, squareform

    ``x`` is an array of five points in three-dimensional space.

    >>> x = np.array([[2, 0, 2], [2, 2, 3], [-2, 4, 5], [0, 1, 9], [2, 2, 4]])

    ``pdist(x)`` computes the Euclidean distances between each pair of
    points in ``x``.  The distances are returned in a one-dimensional
    array with length ``5*(5 - 1)/2 = 10``.

    >>> distvec = pdist(x)
    >>> distvec
    array([2.23606798, 6.40312424, 7.34846923, 2.82842712, 4.89897949,
           6.40312424, 1.        , 5.38516481, 4.58257569, 5.47722558])

    ``squareform(distvec)`` returns the 5x5 distance matrix.

    >>> m = squareform(distvec)
    >>> m
    array([[0.        , 2.23606798, 6.40312424, 7.34846923, 2.82842712],
           [2.23606798, 0.        , 4.89897949, 6.40312424, 1.        ],
           [6.40312424, 4.89897949, 0.        , 5.38516481, 4.58257569],
           [7.34846923, 6.40312424, 5.38516481, 0.        , 5.47722558],
           [2.82842712, 1.        , 4.58257569, 5.47722558, 0.        ]])

    When given a square distance matrix ``m``, ``squareform(m)`` returns
    the one-dimensional condensed distance vector associated with the
    matrix.  In this case, we recover ``distvec``.

    >>> squareform(m)
    array([2.23606798, 6.40312424, 7.34846923, 2.82842712, 4.89897949,
           6.40312424, 1.        , 5.38516481, 4.58257569, 5.47722558])
    """
    X = np.ascontiguousarray(X)
    s = X.shape
    if force.lower() == 'tomatrix':
        if len(s) != 1:
            raise ValueError("Forcing 'tomatrix' but input X is not a distance vector.")
    elif force.lower() == 'tovector':
        if len(s) != 2:
            raise ValueError("Forcing 'tovector' but input X is not a distance matrix.")
    if len(s) == 1:
        if s[0] == 0:
            return np.zeros((1, 1), dtype=X.dtype)
        d = int(np.ceil(np.sqrt(s[0] * 2)))
        if d * (d - 1) != s[0] * 2:
            raise ValueError('Incompatible vector size. It must be a binomial coefficient n choose 2 for some integer n >= 2.')
        M = np.zeros((d, d), dtype=X.dtype)
        X = _copy_array_if_base_present(X)
        _distance_wrap.to_squareform_from_vector_wrap(M, X)
        return M
    elif len(s) == 2:
        if s[0] != s[1]:
            raise ValueError('The matrix argument must be square.')
        if checks:
            is_valid_dm(X, throw=True, name='X')
        d = s[0]
        if d <= 1:
            return np.array([], dtype=X.dtype)
        v = np.zeros(d * (d - 1) // 2, dtype=X.dtype)
        X = _copy_array_if_base_present(X)
        _distance_wrap.to_vector_from_squareform_wrap(X, v)
        return v
    else:
        raise ValueError('The first argument must be one or two dimensional array. A %d-dimensional array is not permitted' % len(s))