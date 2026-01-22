import functools
from math import ceil
import numbers
import scipy.stats
import numpy as np
from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.utils import _supported_float_type, warn
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .. import color
from ..color.colorconv import ycbcr_from_rgb
def denoise_tv_chambolle(image, weight=0.1, eps=0.0002, max_num_iter=200, *, channel_axis=None):
    """Perform total variation denoising in nD.

    Given :math:`f`, a noisy image (input data),
    total variation denoising (also known as total variation regularization)
    aims to find an image :math:`u` with less total variation than :math:`f`,
    under the constraint that :math:`u` remain similar to :math:`f`.
    This can be expressed by the Rudin--Osher--Fatemi (ROF) minimization
    problem:

    .. math::

        \\min_{u} \\sum_{i=0}^{N-1} \\left( \\left| \\nabla{u_i} \\right| + \\frac{\\lambda}{2}(f_i - u_i)^2 \\right)

    where :math:`\\lambda` is a positive parameter.
    The first term of this cost function is the total variation;
    the second term represents data fidelity. As :math:`\\lambda \\to 0`,
    the total variation term dominates, forcing the solution to have smaller
    total variation, at the expense of looking less like the input data.

    This code is an implementation of the algorithm proposed by Chambolle
    in [1]_ to solve the ROF problem.

    Parameters
    ----------
    image : ndarray
        Input image to be denoised. If its dtype is not float, it gets
        converted with :func:`~.img_as_float`.
    weight : float, optional
        Denoising weight. It is equal to :math:`\\frac{1}{\\lambda}`. Therefore,
        the greater the `weight`, the more denoising (at the expense of
        fidelity to `image`).
    eps : float, optional
        Tolerance :math:`\\varepsilon > 0` for the stop criterion (compares to
        absolute value of relative difference of the cost function :math:`E`):
        The algorithm stops when :math:`|E_{n-1} - E_n| < \\varepsilon * E_0`.
    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.
    channel_axis : int or None, optional
        If ``None``, the image is assumed to be grayscale (single-channel).
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    u : ndarray
        Denoised image.

    Notes
    -----
    Make sure to set the `channel_axis` parameter appropriately for color
    images.

    The principle of total variation denoising is explained in [2]_.
    It is about minimizing the total variation of an image,
    which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce cartoon-like images, that is,
    piecewise-constant images.

    See Also
    --------
    denoise_tv_bregman : Perform total variation denoising using split-Bregman
        optimization.

    References
    ----------
    .. [1] A. Chambolle, An algorithm for total variation minimization and
           applications, Journal of Mathematical Imaging and Vision,
           Springer, 2004, 20, 89-97.
    .. [2] https://en.wikipedia.org/wiki/Total_variation_denoising

    Examples
    --------
    2D example on astronaut image:

    >>> from skimage import color, data
    >>> img = color.rgb2gray(data.astronaut())[:50, :50]
    >>> rng = np.random.default_rng()
    >>> img += 0.5 * img.std() * rng.standard_normal(img.shape)
    >>> denoised_img = denoise_tv_chambolle(img, weight=60)

    3D example on synthetic data:

    >>> x, y, z = np.ogrid[0:20, 0:20, 0:20]
    >>> mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    >>> mask = mask.astype(float)
    >>> rng = np.random.default_rng()
    >>> mask += 0.2 * rng.standard_normal(mask.shape)
    >>> res = denoise_tv_chambolle(mask, weight=100)

    """
    im_type = image.dtype
    if not im_type.kind == 'f':
        image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        out = np.zeros_like(image)
        for c in range(image.shape[channel_axis]):
            out[_at(c)] = _denoise_tv_chambolle_nd(image[_at(c)], weight, eps, max_num_iter)
    else:
        out = _denoise_tv_chambolle_nd(image, weight, eps, max_num_iter)
    return out