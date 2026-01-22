import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def variation(a, axis=0, ddof=0):
    """
    Compute the coefficient of variation.

    The coefficient of variation is the standard deviation divided by the
    mean.  This function is equivalent to::

        np.std(x, axis=axis, ddof=ddof) / np.mean(x)

    The default for ``ddof`` is 0, but many definitions of the coefficient
    of variation use the square root of the unbiased sample variance
    for the sample standard deviation, which corresponds to ``ddof=1``.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate the coefficient of variation. Default
        is 0. If None, compute over the whole array `a`.
    ddof : int, optional
        Delta degrees of freedom.  Default is 0.

    Returns
    -------
    variation : ndarray
        The calculated variation along the requested axis.

    Notes
    -----
    For more details about `variation`, see `scipy.stats.variation`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import variation
    >>> a = np.array([2,8,4])
    >>> variation(a)
    0.5345224838248487
    >>> b = np.array([2,8,3,4])
    >>> c = np.ma.masked_array(b, mask=[0,0,1,0])
    >>> variation(c)
    0.5345224838248487

    In the example above, it can be seen that this works the same as
    `scipy.stats.variation` except 'stats.mstats.variation' ignores masked
    array elements.

    """
    a, axis = _chk_asarray(a, axis)
    return a.std(axis, ddof=ddof) / a.mean(axis)