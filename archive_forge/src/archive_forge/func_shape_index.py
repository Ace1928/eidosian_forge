import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD
def shape_index(image, sigma=1, mode='constant', cval=0):
    """Compute the shape index.

    The shape index, as defined by Koenderink & van Doorn [1]_, is a
    single valued measure of local curvature, assuming the image as a 3D plane
    with intensities representing heights.

    It is derived from the eigenvalues of the Hessian, and its
    value ranges from -1 to 1 (and is undefined (=NaN) in *flat* regions),
    with following ranges representing following shapes:

    .. table:: Ranges of the shape index and corresponding shapes.

      ===================  =============
      Interval (s in ...)  Shape
      ===================  =============
      [  -1, -7/8)         Spherical cup
      [-7/8, -5/8)         Through
      [-5/8, -3/8)         Rut
      [-3/8, -1/8)         Saddle rut
      [-1/8, +1/8)         Saddle
      [+1/8, +3/8)         Saddle ridge
      [+3/8, +5/8)         Ridge
      [+5/8, +7/8)         Dome
      [+7/8,   +1]         Spherical cap
      ===================  =============

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used for
        smoothing the input data before Hessian eigen value calculation.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    s : ndarray
        Shape index

    References
    ----------
    .. [1] Koenderink, J. J. & van Doorn, A. J.,
           "Surface shape and curvature scales",
           Image and Vision Computing, 1992, 10, 557-564.
           :DOI:`10.1016/0262-8856(92)90076-F`

    Examples
    --------
    >>> from skimage.feature import shape_index
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> s = shape_index(square, sigma=0.1)
    >>> s
    array([[ nan,  nan, -0.5,  nan,  nan],
           [ nan, -0. ,  nan, -0. ,  nan],
           [-0.5,  nan, -1. ,  nan, -0.5],
           [ nan, -0. ,  nan, -0. ,  nan],
           [ nan,  nan, -0.5,  nan,  nan]])
    """
    H = hessian_matrix(image, sigma=sigma, mode=mode, cval=cval, order='rc', use_gaussian_derivatives=False)
    l1, l2 = hessian_matrix_eigvals(H)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2.0 / np.pi * np.arctan((l2 + l1) / (l2 - l1))