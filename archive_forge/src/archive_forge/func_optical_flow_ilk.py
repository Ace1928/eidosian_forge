from functools import partial
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian as gaussian_filter
from .._shared.utils import _supported_float_type
from ..transform import warp
from ._optical_flow_utils import _coarse_to_fine, _get_warp_points
def optical_flow_ilk(reference_image, moving_image, *, radius=7, num_warp=10, gaussian=False, prefilter=False, dtype=np.float32):
    """Coarse to fine optical flow estimator.

    The iterative Lucas-Kanade (iLK) solver is applied at each level
    of the image pyramid. iLK [1]_ is a fast and robust alternative to
    TVL1 algorithm although less accurate for rendering flat surfaces
    and object boundaries (see [2]_).

    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first grayscale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second grayscale image of the sequence.
    radius : int, optional
        Radius of the window considered around each pixel.
    num_warp : int, optional
        Number of times moving_image is warped.
    gaussian : bool, optional
        If True, a Gaussian kernel is used for the local
        integration. Otherwise, a uniform kernel is used.
    prefilter : bool, optional
        Whether to prefilter the estimated optical flow before each
        image warp. When True, a median filter with window size 3
        along each axis is applied. This helps to remove potential
        outliers.
    dtype : dtype, optional
        Output data type: must be floating point. Single precision
        provides good results and saves memory usage and computation
        time compared to double precision.

    Returns
    -------
    flow : ndarray, shape (reference_image.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    Notes
    -----
    - The implemented algorithm is described in **Table2** of [1]_.
    - Color images are not supported.

    References
    ----------
    .. [1] Le Besnerais, G., & Champagnat, F. (2005, September). Dense
       optical flow by iterative local window registration. In IEEE
       International Conference on Image Processing 2005 (Vol. 1,
       pp. I-137). IEEE. :DOI:`10.1109/ICIP.2005.1529706`
    .. [2] Plyer, A., Le Besnerais, G., & Champagnat,
       F. (2016). Massively parallel Lucas Kanade optical flow for
       real-time video processing applications. Journal of Real-Time
       Image Processing, 11(4), 713-730. :DOI:`10.1007/s11554-014-0423-0`

    Examples
    --------
    >>> from skimage.color import rgb2gray
    >>> from skimage.data import stereo_motorcycle
    >>> from skimage.registration import optical_flow_ilk
    >>> reference_image, moving_image, disp = stereo_motorcycle()
    >>> # --- Convert the images to gray level: color is not supported.
    >>> reference_image = rgb2gray(reference_image)
    >>> moving_image = rgb2gray(moving_image)
    >>> flow = optical_flow_ilk(moving_image, reference_image)

    """
    solver = partial(_ilk, radius=radius, num_warp=num_warp, gaussian=gaussian, prefilter=prefilter)
    if np.dtype(dtype) != _supported_float_type(dtype):
        msg = f"dtype={dtype} is not supported. Try 'float32' or 'float64.'"
        raise ValueError(msg)
    return _coarse_to_fine(reference_image, moving_image, solver, dtype=dtype)