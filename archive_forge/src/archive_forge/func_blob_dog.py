import math
import numpy as np
import scipy.ndimage as ndi
from scipy import spatial
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_nD
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .peak import peak_local_max
def blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=0.5, overlap=0.5, *, threshold_rel=None, exclude_border=False):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Difference of Gaussian (DoG) method [1]_, [2]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        The minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    sigma_ratio : float, optional
        The ratio between the standard deviation of Gaussian Kernels used for
        computing the Difference of Gaussians
    threshold : float or None, optional
        The absolute lower bound for scale space maxima. Local maxima smaller
        than `threshold` are ignored. Reduce this to detect blobs with lower
        intensities. If `threshold_rel` is also specified, whichever threshold
        is larger will be used. If None, `threshold_rel` is used instead.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(dog_space) * threshold_rel``, where ``dog_space`` refers to the
        stack of Difference-of-Gaussian (DoG) images computed internally. This
        should have a value between 0 and 1. If None, `threshold` is used
        instead.
    exclude_border : tuple of ints, int, or False, optional
        If tuple of ints, the length of the tuple must match the input array's
        dimensionality.  Each element of the tuple will exclude peaks from
        within `exclude_border`-pixels of the border of the image along that
        dimension.
        If nonzero int, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If zero or False, peaks are identified regardless of their
        distance from the border.

    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, or 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.

    See also
    --------
    skimage.filters.difference_of_gaussians

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach
    .. [2] Lowe, D. G. "Distinctive Image Features from Scale-Invariant
        Keypoints." International Journal of Computer Vision 60, 91–110 (2004).
        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        :DOI:`10.1023/B:VISI.0000029664.99615.94`

    Examples
    --------
    >>> from skimage import data, feature
    >>> coins = data.coins()
    >>> feature.blob_dog(coins, threshold=.05, min_sigma=10, max_sigma=40)
    array([[128., 155.,  10.],
           [198., 155.,  10.],
           [124., 338.,  10.],
           [127., 102.,  10.],
           [193., 281.,  10.],
           [126., 208.,  10.],
           [267., 115.,  10.],
           [197., 102.,  10.],
           [198., 215.,  10.],
           [123., 279.,  10.],
           [126.,  46.,  10.],
           [259., 247.,  10.],
           [196.,  43.,  10.],
           [ 54., 276.,  10.],
           [267., 358.,  10.],
           [ 58., 100.,  10.],
           [259., 305.,  10.],
           [185., 347.,  16.],
           [261., 174.,  16.],
           [ 46., 336.,  16.],
           [ 54., 217.,  10.],
           [ 55., 157.,  10.],
           [ 57.,  41.,  10.],
           [260.,  47.,  16.]])

    Notes
    -----
    The radius of each blob is approximately :math:`\\sqrt{2}\\sigma` for
    a 2-D image and :math:`\\sqrt{3}\\sigma` for a 3-D image.
    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    scalar_sigma = np.isscalar(max_sigma) and np.isscalar(min_sigma)
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float_dtype)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float_dtype)
    min_sigma = np.asarray(min_sigma, dtype=float_dtype)
    max_sigma = np.asarray(max_sigma, dtype=float_dtype)
    if sigma_ratio <= 1.0:
        raise ValueError('sigma_ratio must be > 1.0')
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))
    sigma_list = np.array([min_sigma * sigma_ratio ** i for i in range(k + 1)])
    dog_image_cube = np.empty(image.shape + (k,), dtype=float_dtype)
    gaussian_previous = gaussian(image, sigma=sigma_list[0], mode='reflect')
    for i, s in enumerate(sigma_list[1:]):
        gaussian_current = gaussian(image, sigma=s, mode='reflect')
        dog_image_cube[..., i] = gaussian_previous - gaussian_current
        gaussian_previous = gaussian_current
    sf = 1 / (sigma_ratio - 1)
    dog_image_cube *= sf
    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(dog_image_cube, threshold_abs=threshold, threshold_rel=threshold_rel, exclude_border=exclude_border, footprint=np.ones((3,) * (image.ndim + 1)))
    if local_maxima.size == 0:
        return np.empty((0, image.ndim + (1 if scalar_sigma else image.ndim)))
    lm = local_maxima.astype(float_dtype)
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]
    if scalar_sigma:
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])
    sigma_dim = sigmas_of_peaks.shape[1]
    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)