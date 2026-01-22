import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, convolve
from .._shared.utils import _supported_float_type, check_nD
from ..restoration.uft import laplacian
from ..util.dtype import img_as_float
def farid_v(image, *, mask=None):
    """Find the vertical edges of an image using the Farid transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Farid edge map.

    Notes
    -----
    The kernel was constructed using the 5-tap weights from [1].

    References
    ----------
    .. [1] Farid, H. and Simoncelli, E. P., "Differentiation of discrete
           multidimensional signals", IEEE Transactions on Image Processing
           13(4): 496-508, 2004. :DOI:`10.1109/TIP.2004.823819`
    """
    check_nD(image, 2)
    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
    else:
        image = img_as_float(image)
    result = convolve(image, VFARID_WEIGHTS)
    return _mask_filter_result(result, mask)