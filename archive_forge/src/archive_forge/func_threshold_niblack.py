import inspect
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ..filters._multiotsu import (
from ..transform import integral_image
from ..util import dtype_limits
from ._sparse import _correlate_sparse, _validate_window_size
def threshold_niblack(image, window_size=15, k=0.2):
    """Applies Niblack local threshold to an array.

    A threshold T is calculated for every pixel in the image using the
    following formula::

        T = m(x,y) - k * s(x,y)

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of parameter k in threshold formula.

    Returns
    -------
    threshold : (M, N[, ...]) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    The Bradley threshold is a particular case of the Niblack
    one, being equivalent to

    >>> from skimage import data
    >>> image = data.page()
    >>> q = 1
    >>> threshold_image = threshold_niblack(image, k=0) * q

    for some value ``q``. By default, Bradley and Roth use ``q=1``.


    References
    ----------
    .. [1] W. Niblack, An introduction to Digital Image Processing,
           Prentice-Hall, 1986.
    .. [2] D. Bradley and G. Roth, "Adaptive thresholding using Integral
           Image", Journal of Graphics Tools 12(2), pp. 13-21, 2007.
           :DOI:`10.1080/2151237X.2007.10129236`

    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> threshold_image = threshold_niblack(image, window_size=7, k=0.1)
    """
    m, s = _mean_std(image, window_size)
    return m - k * s