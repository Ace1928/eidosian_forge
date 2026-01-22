import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD, deprecate_func
from ..util import crop
from ._skeletonize_3d_cy import _compute_thin_image
from ._skeletonize_cy import _fast_skeletonize, _skeletonize_loop, _table_lookup_index
def medial_axis(image, mask=None, return_distance=False, *, rng=None):
    """Compute the medial axis transform of a binary image.

    Parameters
    ----------
    image : binary ndarray, shape (M, N)
        The image of the shape to skeletonize. If this input isn't already a
        binary image, it gets converted into one: In this case, zero values are
        considered background (False), nonzero values are considered
        foreground (True).
    mask : binary ndarray, shape (M, N), optional
        If a mask is given, only those elements in `image` with a true
        value in `mask` are used for computing the medial axis.
    return_distance : bool, optional
        If true, the distance transform is returned as well as the skeleton.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.

        The PRNG determines the order in which pixels are processed for
        tiebreaking.

        .. versionadded:: 0.19

    Returns
    -------
    out : ndarray of bools
        Medial axis transform of the image
    dist : ndarray of ints, optional
        Distance transform of the image (only returned if `return_distance`
        is True)

    See Also
    --------
    skeletonize, thin

    Notes
    -----
    This algorithm computes the medial axis transform of an image
    as the ridges of its distance transform.

    The different steps of the algorithm are as follows
     * A lookup table is used, that assigns 0 or 1 to each configuration of
       the 3x3 binary square, whether the central pixel should be removed
       or kept. We want a point to be removed if it has more than one neighbor
       and if removing it does not change the number of connected components.

     * The distance transform to the background is computed, as well as
       the cornerness of the pixel.

     * The foreground (value of 1) points are ordered by
       the distance transform, then the cornerness.

     * A cython function is called to reduce the image to its skeleton. It
       processes pixels in the order determined at the previous step, and
       removes or maintains a pixel according to the lookup table. Because
       of the ordering, it is possible to process all pixels in only one
       pass.

    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=bool)
    >>> square[1:-1, 2:-2] = 1
    >>> square.view(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> medial_axis(square).view(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """
    global _eight_connect
    if mask is None:
        masked_image = image.astype(bool)
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    center_is_foreground = (np.arange(512) & 2 ** 4).astype(bool)
    table = center_is_foreground & (np.array([ndi.label(_pattern_of(index), _eight_connect)[1] != ndi.label(_pattern_of(index & ~2 ** 4), _eight_connect)[1] for index in range(512)]) | np.array([np.sum(_pattern_of(index)) < 3 for index in range(512)]))
    distance = ndi.distance_transform_edt(masked_image)
    if return_distance:
        store_distance = distance.copy()
    cornerness_table = np.array([9 - np.sum(_pattern_of(index)) for index in range(512)])
    corner_score = _table_lookup(masked_image, cornerness_table)
    i, j = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    result = masked_image.copy()
    distance = distance[result]
    i = np.ascontiguousarray(i[result], dtype=np.intp)
    j = np.ascontiguousarray(j[result], dtype=np.intp)
    result = np.ascontiguousarray(result, np.uint8)
    generator = np.random.default_rng(rng)
    tiebreaker = generator.permutation(np.arange(masked_image.sum()))
    order = np.lexsort((tiebreaker, corner_score[masked_image], distance))
    order = np.ascontiguousarray(order, dtype=np.int32)
    table = np.ascontiguousarray(table, dtype=np.uint8)
    _skeletonize_loop(result, i, j, order, table)
    result = result.astype(bool)
    if mask is not None:
        result[~mask] = image[~mask]
    if return_distance:
        return (result, store_distance)
    else:
        return result