import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD, deprecate_func
from ..util import crop
from ._skeletonize_3d_cy import _compute_thin_image
from ._skeletonize_cy import _fast_skeletonize, _skeletonize_loop, _table_lookup_index
def skeletonize(image, *, method=None):
    """Compute the skeleton of a binary image.

    Thinning is used to reduce each connected component in a binary image
    to a single-pixel wide skeleton.

    Parameters
    ----------
    image : ndarray, 2D or 3D
        An image containing the objects to be skeletonized. Zeros or ``False``
        represent background, nonzero values or ``True`` are foreground.
    method : {'zhang', 'lee'}, optional
        Which algorithm to use. Zhang's algorithm [Zha84]_ only works for
        2D images, and is the default for 2D. Lee's algorithm [Lee94]_
        works for 2D or 3D images and is the default for 3D.

    Returns
    -------
    skeleton : ndarray of bool
        The thinned image.

    See Also
    --------
    medial_axis

    References
    ----------
    .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
           via 3-D medial surface/axis thinning algorithms.
           Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

    .. [Zha84] A fast parallel algorithm for thinning digital patterns,
           T. Y. Zhang and C. Y. Suen, Communications of the ACM,
           March 1984, Volume 27, Number 3.

    Examples
    --------
    >>> X, Y = np.ogrid[0:9, 0:9]
    >>> ellipse = (1./3 * (X - 4)**2 + (Y - 4)**2 < 3**2).astype(bool)
    >>> ellipse.view(np.uint8)
    array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)
    >>> skel = skeletonize(ellipse)
    >>> skel.view(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """
    image = image.astype(bool, order='C', copy=False)
    if method not in {'zhang', 'lee', None}:
        raise ValueError(f'skeletonize method should be either "lee" or "zhang", got {method}.')
    if image.ndim == 2 and (method is None or method == 'zhang'):
        skeleton = _skeletonize_2d(image)
    elif image.ndim == 3 and method == 'zhang':
        raise ValueError('skeletonize method "zhang" only works for 2D images.')
    elif image.ndim == 3 or (image.ndim == 2 and method == 'lee'):
        skeleton = _skeletonize_3d(image)
    else:
        raise ValueError(f'skeletonize requires a 2D or 3D image as input, got {image.ndim}D.')
    return skeleton