import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def hermeweight(x):
    """Weight function of the Hermite_e polynomials.

    The weight function is :math:`\\exp(-x^2/2)` and the interval of
    integration is :math:`[-\\inf, \\inf]`. the HermiteE polynomials are
    orthogonal, but not normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    w = np.exp(-0.5 * x ** 2)
    return w