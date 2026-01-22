import math
import numbers
import numpy as np
from .._shared.utils import _supported_float_type
from ..color.adapt_rgb import adapt_rgb, hsv_value
from .exposure import rescale_intensity
from ..util import img_as_uint
@adapt_rgb(hsv_value)
def equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).

    An algorithm for local contrast enhancement, that uses histograms computed
    over different tile regions of the image. Local details can therefore be
    enhanced even in regions that are darker or lighter than most of the image.

    Parameters
    ----------
    image : (M[, ...][, C]) ndarray
        Input image.
    kernel_size : int or array_like, optional
        Defines the shape of contextual regions used in the algorithm. If
        iterable is passed, it must have the same number of elements as
        ``image.ndim`` (without color channel). If integer, it is broadcasted
        to each `image` dimension. By default, ``kernel_size`` is 1/8 of
        ``image`` height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").

    Returns
    -------
    out : (M[, ...][, C]) ndarray
        Equalized image with float64 dtype.

    See Also
    --------
    equalize_hist, rescale_intensity

    Notes
    -----
    * For color images, the following steps are performed:
       - The image is converted to HSV color space
       - The CLAHE algorithm is run on the V (Value) channel
       - The image is converted back to RGB space and returned
    * For RGBA images, the original alpha channel is removed.

    .. versionchanged:: 0.17
        The values returned by this function are slightly shifted upwards
        because of an internal change in rounding behavior.

    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """
    float_dtype = _supported_float_type(image.dtype)
    image = img_as_uint(image)
    image = np.round(rescale_intensity(image, out_range=(0, NR_OF_GRAY - 1))).astype(np.min_scalar_type(NR_OF_GRAY))
    if kernel_size is None:
        kernel_size = tuple([max(s // 8, 1) for s in image.shape])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        raise ValueError(f'Incorrect value of `kernel_size`: {kernel_size}')
    kernel_size = [int(k) for k in kernel_size]
    image = _clahe(image, kernel_size, clip_limit, nbins)
    image = image.astype(float_dtype, copy=False)
    return rescale_intensity(image)