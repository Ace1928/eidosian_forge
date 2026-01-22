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
def trimmed_std(a, limits=(0.1, 0.1), inclusive=(1, 1), relative=True, axis=None, ddof=0):
    """Returns the trimmed standard deviation of the data along the given axis.

    %s
    ddof : {0,integer}, optional
        Means Delta Degrees of Freedom. The denominator used during computations
        is (n-ddof). DDOF=0 corresponds to a biased estimate, DDOF=1 to an un-
        biased estimate of the variance.

    """
    if not isinstance(limits, tuple) and isinstance(limits, float):
        limits = (limits, limits)
    if relative:
        out = trimr(a, limits=limits, inclusive=inclusive, axis=axis)
    else:
        out = trima(a, limits=limits, inclusive=inclusive)
    return out.std(axis=axis, ddof=ddof)