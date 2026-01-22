import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, convolve
from .._shared.utils import _supported_float_type, check_nD
from ..restoration.uft import laplacian
from ..util.dtype import img_as_float
def scharr(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    """Find the edge magnitude using the Scharr transform.

    Parameters
    ----------
    image : array
        The input image.
    mask : array of bool, optional
        Clip the output image to this mask. (Values where mask=0 will be set
        to 0.)
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::

            sch_mag = np.sqrt(sum([scharr(image, axis=i)**2
                                   for i in range(image.ndim)]) / image.ndim)

        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.

    Returns
    -------
    output : array of float
        The Scharr edge map.

    See also
    --------
    scharr_h, scharr_v : horizontal and vertical edge detection.
    sobel, prewitt, farid, skimage.feature.canny

    Notes
    -----
    The Scharr operator has a better rotation invariance than
    other edge filters such as the Sobel or the Prewitt operators.

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical
           Optimization of Kernel Based Image Derivatives.

    .. [2] https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators

    Examples
    --------
    >>> from skimage import data
    >>> from skimage import filters
    >>> camera = data.camera()
    >>> edges = filters.scharr(camera)
    """
    output = _generic_edge_filter(image, smooth_weights=SCHARR_SMOOTH, axis=axis, mode=mode, cval=cval)
    output = _mask_filter_result(output, mask)
    return output