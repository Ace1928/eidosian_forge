import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def chebpts2(npts):
    """
    Chebyshev points of the second kind.

    The Chebyshev points of the second kind are the points ``cos(x)``,
    where ``x = [pi*k/(npts - 1) for k in range(npts)]`` sorted in ascending
    order.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the second kind.

    Notes
    -----

    .. versionadded:: 1.5.0

    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError('npts must be integer')
    if _npts < 2:
        raise ValueError('npts must be >= 2')
    x = np.linspace(-np.pi, 0, _npts)
    return np.cos(x)