import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def equalize_hist(image, nbins=256, mask=None):
    """Return image after histogram equalization.

    Parameters
    ----------
    image : array
        Image array.
    nbins : int, optional
        Number of bins for image histogram. Note: this argument is
        ignored for integer images, for which each integer is its own
        bin.
    mask : ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        are used for the equalization, which is applied to the whole image.

    Returns
    -------
    out : float array
        Image array after histogram equalization.

    Notes
    -----
    This function is adapted from [1]_ with the author's permission.

    References
    ----------
    .. [1] http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    .. [2] https://en.wikipedia.org/wiki/Histogram_equalization

    """
    if mask is not None:
        mask = np.array(mask, dtype=bool)
        cdf, bin_centers = cumulative_distribution(image[mask], nbins)
    else:
        cdf, bin_centers = cumulative_distribution(image, nbins)
    out = np.interp(image.flat, bin_centers, cdf)
    out = out.reshape(image.shape)
    return out.astype(utils._supported_float_type(image.dtype), copy=False)