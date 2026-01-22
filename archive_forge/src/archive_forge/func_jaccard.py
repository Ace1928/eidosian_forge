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
def jaccard(u, v, w=None):
    """
    Compute the Jaccard-Needham dissimilarity between two boolean 1-D arrays.

    The Jaccard-Needham dissimilarity between 1-D boolean arrays `u` and `v`,
    is defined as

    .. math::

       \\frac{c_{TF} + c_{FT}}
            {c_{TT} + c_{FT} + c_{TF}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    jaccard : double
        The Jaccard distance between vectors `u` and `v`.

    Notes
    -----
    When both `u` and `v` lead to a `0/0` division i.e. there is no overlap
    between the items in the vectors the returned distance is 0. See the
    Wikipedia page on the Jaccard index [1]_, and this paper [2]_.

    .. versionchanged:: 1.2.0
        Previously, when `u` and `v` lead to a `0/0` division, the function
        would return NaN. This was changed to return 0 instead.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Jaccard_index
    .. [2] S. Kosub, "A note on the triangle inequality for the Jaccard
       distance", 2016, :arxiv:`1612.02696`

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.jaccard([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.jaccard([1, 0, 0], [1, 1, 0])
    0.5
    >>> distance.jaccard([1, 0, 0], [1, 2, 0])
    0.5
    >>> distance.jaccard([1, 0, 0], [1, 1, 1])
    0.66666666666666663

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    nonzero = np.bitwise_or(u != 0, v != 0)
    unequal_nonzero = np.bitwise_and(u != v, nonzero)
    if w is not None:
        w = _validate_weights(w)
        nonzero = w * nonzero
        unequal_nonzero = w * unequal_nonzero
    a = np.float64(unequal_nonzero.sum())
    b = np.float64(nonzero.sum())
    return a / b if b != 0 else 0