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
def trimtail(data, proportiontocut=0.2, tail='left', inclusive=(True, True), axis=None):
    """
    Trims the data by masking values from one tail.

    Parameters
    ----------
    data : array_like
        Data to trim.
    proportiontocut : float, optional
        Percentage of trimming. If n is the number of unmasked values
        before trimming, the number of values after trimming is
        ``(1 - proportiontocut) * n``.  Default is 0.2.
    tail : {'left','right'}, optional
        If 'left' the `proportiontocut` lowest values will be masked.
        If 'right' the `proportiontocut` highest values will be masked.
        Default is 'left'.
    inclusive : {(bool, bool) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).  Default is
        (True, True).
    axis : int, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.  Default is None.

    Returns
    -------
    trimtail : ndarray
        Returned array of same shape as `data` with masked tail values.

    """
    tail = str(tail).lower()[0]
    if tail == 'l':
        limits = (proportiontocut, None)
    elif tail == 'r':
        limits = (None, proportiontocut)
    else:
        raise TypeError("The tail argument should be in ('left','right')")
    return trimr(data, limits=limits, axis=axis, inclusive=inclusive)