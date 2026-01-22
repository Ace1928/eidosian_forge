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
def hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc', use_gaussian_derivatives=None):
    """Compute the Hessian matrix.

    In 2D, the Hessian matrix is defined as::

        H = [Hrr Hrc]
            [Hrc Hcc]

    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective r- and c-directions.

    The implementation here also supports n-dimensional data.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        For 2D images, this parameter allows for the use of reverse or forward
        order of the image axes in gradient computation. 'rc' indicates the use
        of the first axis initially (Hrr, Hrc, Hcc), whilst 'xy' indicates the
        usage of the last axis initially (Hxx, Hxy, Hyy). Images with higher
        dimension must always use 'rc' order.
    use_gaussian_derivatives : boolean, optional
        Indicates whether the Hessian is computed by convolving with Gaussian
        derivatives, or by a simple finite-difference operation.

    Returns
    -------
    H_elems : list of ndarray
        Upper-diagonal elements of the hessian matrix for each pixel in the
        input image. In 2D, this will be a three element list containing [Hrr,
        Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.


    Notes
    -----
    The distributive property of derivatives and convolutions allows us to
    restate the derivative of an image, I, smoothed with a Gaussian kernel, G,
    as the convolution of the image with the derivative of G.

    .. math::

        \\frac{\\partial }{\\partial x_i}(I * G) =
        I * \\left( \\frac{\\partial }{\\partial x_i} G \\right)

    When ``use_gaussian_derivatives`` is ``True``, this property is used to
    compute the second order derivatives that make up the Hessian matrix.

    When ``use_gaussian_derivatives`` is ``False``, simple finite differences
    on a Gaussian-smoothed image are used instead.

    Examples
    --------
    >>> from skimage.feature import hessian_matrix
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc',
    ...                                use_gaussian_derivatives=False)
    >>> Hrc
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == 'xy':
        raise ValueError("order='xy' is only supported for 2D images.")
    if order not in ['rc', 'xy']:
        raise ValueError(f'unrecognized order: {order}')
    if use_gaussian_derivatives is None:
        use_gaussian_derivatives = False
        warn('use_gaussian_derivatives currently defaults to False, but will change to True in a future version. Please specify this argument explicitly to maintain the current behavior', category=FutureWarning, stacklevel=2)
    if use_gaussian_derivatives:
        return _hessian_matrix_with_gaussian(image, sigma=sigma, mode=mode, cval=cval, order=order)
    gaussian_filtered = gaussian(image, sigma=sigma, mode=mode, cval=cval)
    gradients = np.gradient(gaussian_filtered)
    axes = range(image.ndim)
    if order == 'xy':
        axes = reversed(axes)
    H_elems = [np.gradient(gradients[ax0], axis=ax1) for ax0, ax1 in combinations_with_replacement(axes, 2)]
    return H_elems