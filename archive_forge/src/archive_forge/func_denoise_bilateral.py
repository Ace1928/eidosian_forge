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
@utils.channel_as_last_axis()
def denoise_bilateral(image, win_size=None, sigma_color=None, sigma_spatial=1, bins=10000, mode='constant', cval=0, *, channel_axis=None):
    """Denoise image using bilateral filter.

    Parameters
    ----------
    image : ndarray, shape (M, N[, 3])
        Input image, 2D grayscale or RGB.
    win_size : int
        Window size for filtering.
        If win_size is not specified, it is calculated as
        ``max(5, 2 * ceil(3 * sigma_spatial) + 1)``.
    sigma_color : float
        Standard deviation for grayvalue/color distance (radiometric
        similarity). A larger value results in averaging of pixels with larger
        radiometric differences. If ``None``, the standard deviation of
        ``image`` will be used.
    sigma_spatial : float
        Standard deviation for range distance. A larger value results in
        averaging of pixels with larger spatial differences.
    bins : int
        Number of discrete values for Gaussian weights of color filtering.
        A larger value results in improved accuracy.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
        How to handle values outside the image borders. See
        `numpy.pad` for detail.
    cval : int or float
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    channel_axis : int or None, optional
        If ``None``, the image is assumed to be grayscale (single-channel).
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    denoised : ndarray
        Denoised image.

    Notes
    -----
    This is an edge-preserving, denoising filter. It averages pixels based on
    their spatial closeness and radiometric similarity [1]_.

    Spatial closeness is measured by the Gaussian function of the Euclidean
    distance between two pixels and a certain standard deviation
    (`sigma_spatial`).

    Radiometric similarity is measured by the Gaussian function of the
    Euclidean distance between two color values and a certain standard
    deviation (`sigma_color`).

    Note that, if the image is of any `int` dtype, ``image`` will be
    converted using the `img_as_float` function and thus the standard
    deviation (`sigma_color`) will be in range ``[0, 1]``.

    For more information on scikit-image's data type conversions and how
    images are rescaled in these conversions,
    see: https://scikit-image.org/docs/stable/user_guide/data_types.html.

    References
    ----------
    .. [1] C. Tomasi and R. Manduchi. "Bilateral Filtering for Gray and Color
           Images." IEEE International Conference on Computer Vision (1998)
           839-846. :DOI:`10.1109/ICCV.1998.710815`

    Examples
    --------
    >>> from skimage import data, img_as_float
    >>> astro = img_as_float(data.astronaut())
    >>> astro = astro[220:300, 220:320]
    >>> rng = np.random.default_rng()
    >>> noisy = astro + 0.6 * astro.std() * rng.random(astro.shape)
    >>> noisy = np.clip(noisy, 0, 1)
    >>> denoised = denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,
    ...                              channel_axis=-1)
    """
    if channel_axis is not None:
        if image.ndim != 3:
            if image.ndim == 2:
                raise ValueError('Use ``channel_axis=None`` for 2D grayscale images. The last axis of the input image must be multiple color channels not another spatial dimension.')
            else:
                raise ValueError(f'Bilateral filter is only implemented for 2D grayscale images (image.ndim == 2) and 2D multichannel (image.ndim == 3) images, but the input image has {image.ndim} dimensions.')
        elif image.shape[2] not in (3, 4):
            if image.shape[2] > 4:
                msg = f'The last axis of the input image is interpreted as channels. Input image with shape {image.shape} has {image.shape[2]} channels in last axis. ``denoise_bilateral``is implemented for 2D grayscale and color images only.'
                warn(msg)
            else:
                msg = f'Input image must be grayscale, RGB, or RGBA; but has shape {image.shape}.'
                warn(msg)
    elif image.ndim > 2:
        raise ValueError(f'Bilateral filter is not implemented for grayscale images of 3 or more dimensions, but input image has {image.shape} shape. Use ``channel_axis=-1`` for 2D RGB images.')
    if win_size is None:
        win_size = max(5, 2 * int(ceil(3 * sigma_spatial)) + 1)
    min_value = image.min()
    max_value = image.max()
    if min_value == max_value:
        return image
    image = np.atleast_3d(img_as_float(image))
    image = np.ascontiguousarray(image)
    sigma_color = sigma_color or image.std()
    color_lut = _compute_color_lut(bins, sigma_color, max_value, dtype=image.dtype)
    range_lut = _compute_spatial_lut(win_size, sigma_spatial, dtype=image.dtype)
    out = np.empty(image.shape, dtype=image.dtype)
    dims = image.shape[2]
    empty_dims = np.empty(dims, dtype=image.dtype)
    if min_value < 0:
        image = image - min_value
        max_value -= min_value
    _denoise_bilateral(image, max_value, win_size, sigma_color, sigma_spatial, bins, mode, cval, color_lut, range_lut, empty_dims, out)
    out = np.squeeze(out)
    if min_value < 0:
        out += min_value
    return out