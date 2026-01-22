import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def compare_medians_ms(group_1, group_2, axis=None):
    """
    Compares the medians from two independent groups along the given axis.

    The comparison is performed using the McKean-Schrader estimate of the
    standard error of the medians.

    Parameters
    ----------
    group_1 : array_like
        First dataset.  Has to be of size >=7.
    group_2 : array_like
        Second dataset.  Has to be of size >=7.
    axis : int, optional
        Axis along which the medians are estimated. If None, the arrays are
        flattened.  If `axis` is not None, then `group_1` and `group_2`
        should have the same shape.

    Returns
    -------
    compare_medians_ms : {float, ndarray}
        If `axis` is None, then returns a float, otherwise returns a 1-D
        ndarray of floats with a length equal to the length of `group_1`
        along `axis`.

    Examples
    --------

    >>> from scipy import stats
    >>> a = [1, 2, 3, 4, 5, 6, 7]
    >>> b = [8, 9, 10, 11, 12, 13, 14]
    >>> stats.mstats.compare_medians_ms(a, b, axis=None)
    1.0693225866553746e-05

    The function is vectorized to compute along a given axis.

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.random(size=(3, 7))
    >>> y = rng.random(size=(3, 8))
    >>> stats.mstats.compare_medians_ms(x, y, axis=1)
    array([0.36908985, 0.36092538, 0.2765313 ])

    References
    ----------
    .. [1] McKean, Joseph W., and Ronald M. Schrader. "A comparison of methods
       for studentizing the sample median." Communications in
       Statistics-Simulation and Computation 13.6 (1984): 751-773.

    """
    med_1, med_2 = (ma.median(group_1, axis=axis), ma.median(group_2, axis=axis))
    std_1, std_2 = (mstats.stde_median(group_1, axis=axis), mstats.stde_median(group_2, axis=axis))
    W = np.abs(med_1 - med_2) / ma.sqrt(std_1 ** 2 + std_2 ** 2)
    return 1 - norm.cdf(W)