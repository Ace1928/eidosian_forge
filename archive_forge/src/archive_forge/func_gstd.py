import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
def gstd(a, axis=0, ddof=1):
    """
    Calculate the geometric standard deviation of an array.

    The geometric standard deviation describes the spread of a set of numbers
    where the geometric mean is preferred. It is a multiplicative factor, and
    so a dimensionless quantity.

    It is defined as the exponent of the standard deviation of ``log(a)``.
    Mathematically the population geometric standard deviation can be
    evaluated as::

        gstd = exp(std(log(a)))

    .. versionadded:: 1.3.0

    Parameters
    ----------
    a : array_like
        An array like object containing the sample data.
    axis : int, tuple or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    ddof : int, optional
        Degree of freedom correction in the calculation of the
        geometric standard deviation. Default is 1.

    Returns
    -------
    gstd : ndarray or float
        An array of the geometric standard deviation. If `axis` is None or `a`
        is a 1d array a float is returned.

    See Also
    --------
    gmean : Geometric mean
    numpy.std : Standard deviation
    gzscore : Geometric standard score

    Notes
    -----
    As the calculation requires the use of logarithms the geometric standard
    deviation only supports strictly positive values. Any non-positive or
    infinite values will raise a `ValueError`.
    The geometric standard deviation is sometimes confused with the exponent of
    the standard deviation, ``exp(std(a))``. Instead the geometric standard
    deviation is ``exp(std(log(a)))``.
    The default value for `ddof` is different to the default value (0) used
    by other ddof containing functions, such as ``np.std`` and ``np.nanstd``.

    References
    ----------
    .. [1] "Geometric standard deviation", *Wikipedia*,
           https://en.wikipedia.org/wiki/Geometric_standard_deviation.
    .. [2] Kirkwood, T. B., "Geometric means and measures of dispersion",
           Biometrics, vol. 35, pp. 908-909, 1979

    Examples
    --------
    Find the geometric standard deviation of a log-normally distributed sample.
    Note that the standard deviation of the distribution is one, on a
    log scale this evaluates to approximately ``exp(1)``.

    >>> import numpy as np
    >>> from scipy.stats import gstd
    >>> rng = np.random.default_rng()
    >>> sample = rng.lognormal(mean=0, sigma=1, size=1000)
    >>> gstd(sample)
    2.810010162475324

    Compute the geometric standard deviation of a multidimensional array and
    of a given axis.

    >>> a = np.arange(1, 25).reshape(2, 3, 4)
    >>> gstd(a, axis=None)
    2.2944076136018947
    >>> gstd(a, axis=2)
    array([[1.82424757, 1.22436866, 1.13183117],
           [1.09348306, 1.07244798, 1.05914985]])
    >>> gstd(a, axis=(1,2))
    array([2.12939215, 1.22120169])

    The geometric standard deviation further handles masked arrays.

    >>> a = np.arange(1, 25).reshape(2, 3, 4)
    >>> ma = np.ma.masked_where(a > 16, a)
    >>> ma
    masked_array(
      data=[[[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]],
            [[13, 14, 15, 16],
             [--, --, --, --],
             [--, --, --, --]]],
      mask=[[[False, False, False, False],
             [False, False, False, False],
             [False, False, False, False]],
            [[False, False, False, False],
             [ True,  True,  True,  True],
             [ True,  True,  True,  True]]],
      fill_value=999999)
    >>> gstd(ma, axis=2)
    masked_array(
      data=[[1.8242475707663655, 1.2243686572447428, 1.1318311657788478],
            [1.0934830582350938, --, --]],
      mask=[[False, False, False],
            [False,  True,  True]],
      fill_value=999999)

    """
    a = np.asanyarray(a)
    log = ma.log if isinstance(a, ma.MaskedArray) else np.log
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            return np.exp(np.std(log(a), axis=axis, ddof=ddof))
    except RuntimeWarning as w:
        if np.isinf(a).any():
            raise ValueError('Infinite value encountered. The geometric standard deviation is defined for strictly positive values only.') from w
        a_nan = np.isnan(a)
        a_nan_any = a_nan.any()
        if a_nan_any and np.less_equal(np.nanmin(a), 0) or (not a_nan_any and np.less_equal(a, 0).any()):
            raise ValueError('Non positive value encountered. The geometric standard deviation is defined for strictly positive values only.') from w
        elif 'Degrees of freedom <= 0 for slice' == str(w):
            raise ValueError(w) from w
        else:
            return np.exp(np.std(log(a, where=~a_nan), axis=axis, ddof=ddof))
    except TypeError as e:
        raise ValueError('Invalid array input. The inputs could not be safely coerced to any supported types') from e