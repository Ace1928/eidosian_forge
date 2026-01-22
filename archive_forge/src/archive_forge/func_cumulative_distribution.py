import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def cumulative_distribution(image, nbins=256):
    """Return cumulative distribution function (cdf) for the given image.

    Parameters
    ----------
    image : array
        Image array.
    nbins : int, optional
        Number of bins for image histogram.

    Returns
    -------
    img_cdf : array
        Values of cumulative distribution function.
    bin_centers : array
        Centers of bins.

    See Also
    --------
    histogram

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cumulative_distribution_function

    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.camera())
    >>> hi = exposure.histogram(image)
    >>> cdf = exposure.cumulative_distribution(image)
    >>> all(cdf[0] == np.cumsum(hi[0])/float(image.size))
    True
    """
    hist, bin_centers = histogram(image, nbins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    cdf_dtype = utils._supported_float_type(image.dtype)
    img_cdf = img_cdf.astype(cdf_dtype, copy=False)
    return (img_cdf, bin_centers)