import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def get_bins(values):
    """
    Automatically compute the number of bins for discrete variables.

    Parameters
    ----------
    values = numpy array
        values

    Returns
    -------
    array with the bins

    Notes
    -----
    Computes the width of the bins by taking the maximum of the Sturges and the Freedman-Diaconis
    estimators. According to numpy `np.histogram` this provides good all around performance.

    The Sturges is a very simplistic estimator based on the assumption of normality of the data.
    This estimator has poor performance for non-normal data, which becomes especially obvious for
    large data sets. The estimate depends only on size of the data.

    The Freedman-Diaconis rule uses interquartile range (IQR) to estimate the binwidth.
    It is considered a robust version of the Scott rule as the IQR is less affected by outliers
    than the standard deviation. However, the IQR depends on fewer points than the standard
    deviation, so it is less accurate, especially for long tailed distributions.
    """
    dtype = values.dtype.kind
    if dtype == 'i':
        x_min = values.min().astype(int)
        x_max = values.max().astype(int)
    else:
        x_min = values.min().astype(float)
        x_max = values.max().astype(float)
    bins_sturges = (x_max - x_min) / (np.log2(values.size) + 1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    bins_fd = 2 * iqr * values.size ** (-1 / 3)
    if dtype == 'i':
        width = np.round(np.max([1, bins_sturges, bins_fd])).astype(int)
        bins = np.arange(x_min, x_max + width + 1, width)
    else:
        width = np.max([bins_sturges, bins_fd])
        if np.isclose(x_min, x_max):
            width = 0.001
        bins = np.arange(x_min, x_max + width, width)
    return bins