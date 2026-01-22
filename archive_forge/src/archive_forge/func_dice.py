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
def dice(u, v, w=None):
    """
    Compute the Dice dissimilarity between two boolean 1-D arrays.

    The Dice dissimilarity between `u` and `v`, is

    .. math::

         \\frac{c_{TF} + c_{FT}}
              {2c_{TT} + c_{FT} + c_{TF}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input 1-D array.
    v : (N,) array_like, bool
        Input 1-D array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    dice : double
        The Dice dissimilarity between 1-D arrays `u` and `v`.

    Notes
    -----
    This function computes the Dice dissimilarity index. To compute the
    Dice similarity index, convert one to the other with similarity =
    1 - dissimilarity.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.dice([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.dice([1, 0, 0], [1, 1, 0])
    0.3333333333333333
    >>> distance.dice([1, 0, 0], [2, 0, 0])
    -0.3333333333333333

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if w is not None:
        w = _validate_weights(w)
    if u.dtype == v.dtype == bool and w is None:
        ntt = (u & v).sum()
    else:
        dtype = np.result_type(int, u.dtype, v.dtype)
        u = u.astype(dtype)
        v = v.astype(dtype)
        if w is None:
            ntt = (u * v).sum()
        else:
            ntt = (u * v * w).sum()
    nft, ntf = _nbool_correspond_ft_tf(u, v, w=w)
    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))