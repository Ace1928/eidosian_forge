from statsmodels.compat.numpy import NP_LT_123
import numpy as np
from scipy.special import comb
from statsmodels.graphics.utils import _import_mpl
from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import itertools
from multiprocessing import Pool
from . import utils
def banddepth(data, method='MBD'):
    """
    Calculate the band depth for a set of functional curves.

    Band depth is an order statistic for functional data (see `fboxplot`), with
    a higher band depth indicating larger "centrality".  In analog to scalar
    data, the functional curve with highest band depth is called the median
    curve, and the band made up from the first N/2 of N curves is the 50%
    central region.

    Parameters
    ----------
    data : ndarray
        The vectors of functions to create a functional boxplot from.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    method : {'MBD', 'BD2'}, optional
        Whether to use the original band depth (with J=2) of [1]_ or the
        modified band depth.  See Notes for details.

    Returns
    -------
    ndarray
        Depth values for functional curves.

    Notes
    -----
    Functional band depth as an order statistic for functional data was
    proposed in [1]_ and applied to functional boxplots and bagplots in [2]_.

    The method 'BD2' checks for each curve whether it lies completely inside
    bands constructed from two curves.  All permutations of two curves in the
    set of curves are used, and the band depth is normalized to one.  Due to
    the complete curve having to fall within the band, this method yields a lot
    of ties.

    The method 'MBD' is similar to 'BD2', but checks the fraction of the curve
    falling within the bands.  It therefore generates very few ties.

    The algorithm uses the efficient implementation proposed in [3]_.

    References
    ----------
    .. [1] S. Lopez-Pintado and J. Romo, "On the Concept of Depth for
           Functional Data", Journal of the American Statistical Association,
           vol.  104, pp. 718-734, 2009.
    .. [2] Y. Sun and M.G. Genton, "Functional Boxplots", Journal of
           Computational and Graphical Statistics, vol. 20, pp. 1-19, 2011.
    .. [3] Y. Sun, M. G. Gentonb and D. W. Nychkac, "Exact fast computation
           of band depth for large functional datasets: How quickly can one
           million curves be ranked?", Journal for the Rapid Dissemination
           of Statistics Research, vol. 1, pp. 68-74, 2012.
    """
    n, p = data.shape
    rv = np.argsort(data, axis=0)
    rmat = np.argsort(rv, axis=0) + 1

    def _fbd2():
        down = np.min(rmat, axis=1) - 1
        up = n - np.max(rmat, axis=1)
        return (up * down + n - 1) / comb(n, 2)

    def _fmbd():
        down = rmat - 1
        up = n - rmat
        return (np.sum(up * down, axis=1) / p + n - 1) / comb(n, 2)
    if method == 'BD2':
        depth = _fbd2()
    elif method == 'MBD':
        depth = _fmbd()
    else:
        raise ValueError('Unknown input value for parameter `method`.')
    return depth