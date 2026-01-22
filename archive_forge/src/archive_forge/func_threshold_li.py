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
def threshold_li(image, *, tolerance=None, initial_guess=None, iter_callback=None):
    """Compute threshold value by Li's iterative Minimum Cross Entropy method.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    tolerance : float, optional
        Finish the computation when the change in the threshold in an iteration
        is less than this value. By default, this is half the smallest
        difference between intensity values in ``image``.
    initial_guess : float or Callable[[array[float]], float], optional
        Li's iterative method uses gradient descent to find the optimal
        threshold. If the image intensity histogram contains more than two
        modes (peaks), the gradient descent could get stuck in a local optimum.
        An initial guess for the iteration can help the algorithm find the
        globally-optimal threshold. A float value defines a specific start
        point, while a callable should take in an array of image intensities
        and return a float value. Example valid callables include
        ``numpy.mean`` (default), ``lambda arr: numpy.quantile(arr, 0.95)``,
        or even :func:`skimage.filters.threshold_otsu`.
    iter_callback : Callable[[float], Any], optional
        A function that will be called on the threshold at every iteration of
        the algorithm.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           :DOI:`10.1016/0031-3203(93)90115-D`
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
           :DOI:`10.1016/S0167-8655(98)00057-9`
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           :DOI:`10.1117/1.1631315`
    .. [4] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_li(image)
    >>> binary = image > thresh
    """
    image = image[~np.isnan(image)]
    if image.size == 0:
        return np.nan
    if np.all(image == image.flat[0]):
        return image.flat[0]
    image = image[np.isfinite(image)]
    if image.size == 0:
        return 0.0
    image_min = np.min(image)
    image -= image_min
    if image.dtype.kind in 'iu':
        tolerance = tolerance or 0.5
    else:
        tolerance = tolerance or np.min(np.diff(np.unique(image))) / 2
    if initial_guess is None:
        t_next = np.mean(image)
    elif callable(initial_guess):
        t_next = initial_guess(image)
    elif np.isscalar(initial_guess):
        t_next = initial_guess - float(image_min)
        image_max = np.max(image) + image_min
        if not 0 < t_next < np.max(image):
            msg = f'The initial guess for threshold_li must be within the range of the image. Got {initial_guess} for image min {image_min} and max {image_max}.'
            raise ValueError(msg)
        t_next = image.dtype.type(t_next)
    else:
        raise TypeError('Incorrect type for `initial_guess`; should be a floating point value, or a function mapping an array to a floating point value.')
    t_curr = -2 * tolerance
    if iter_callback is not None:
        iter_callback(t_next + image_min)
    if image.dtype.kind in 'iu':
        hist, bin_centers = histogram(image.reshape(-1), source_range='image')
        hist = hist.astype('float32', copy=False)
        while abs(t_next - t_curr) > tolerance:
            t_curr = t_next
            foreground = bin_centers > t_curr
            background = ~foreground
            mean_fore = np.average(bin_centers[foreground], weights=hist[foreground])
            mean_back = np.average(bin_centers[background], weights=hist[background])
            if mean_back == 0:
                break
            t_next = (mean_back - mean_fore) / (np.log(mean_back) - np.log(mean_fore))
            if iter_callback is not None:
                iter_callback(t_next + image_min)
    else:
        while abs(t_next - t_curr) > tolerance:
            t_curr = t_next
            foreground = image > t_curr
            mean_fore = np.mean(image[foreground])
            mean_back = np.mean(image[~foreground])
            if mean_back == 0.0:
                break
            t_next = (mean_back - mean_fore) / (np.log(mean_back) - np.log(mean_fore))
            if iter_callback is not None:
                iter_callback(t_next + image_min)
    threshold = t_next + image_min
    return threshold