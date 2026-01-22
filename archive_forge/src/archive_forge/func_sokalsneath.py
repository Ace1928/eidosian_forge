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
def sokalsneath(u, v, w=None):
    """
    Compute the Sokal-Sneath dissimilarity between two boolean 1-D arrays.

    The Sokal-Sneath dissimilarity between `u` and `v`,

    .. math::

       \\frac{R}
            {c_{TT} + R}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n` and :math:`R = 2(c_{TF} + c_{FT})`.

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
    sokalsneath : double
        The Sokal-Sneath dissimilarity between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.sokalsneath([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.sokalsneath([1, 0, 0], [1, 1, 0])
    0.66666666666666663
    >>> distance.sokalsneath([1, 0, 0], [2, 1, 0])
    0.0
    >>> distance.sokalsneath([1, 0, 0], [3, 1, 0])
    -2.0

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if u.dtype == v.dtype == bool and w is None:
        ntt = (u & v).sum()
    elif w is None:
        ntt = (u * v).sum()
    else:
        w = _validate_weights(w)
        ntt = (u * v * w).sum()
    nft, ntf = _nbool_correspond_ft_tf(u, v, w=w)
    denom = np.array(ntt + 2.0 * (ntf + nft))
    if not denom.any():
        raise ValueError('Sokal-Sneath dissimilarity is not defined for vectors that are entirely false.')
    return float(2.0 * (ntf + nft)) / denom