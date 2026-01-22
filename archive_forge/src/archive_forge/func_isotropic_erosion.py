import numpy as np
from scipy import ndimage as ndi
def isotropic_erosion(image, radius, out=None, spacing=None):
    """Return binary morphological erosion of an image.

    This function returns the same result as :func:`skimage.morphology.binary_erosion`
    but performs faster for large circular structuring elements.
    This works by applying a threshold to the exact Euclidean distance map
    of the image [1]_, [2]_.
    The implementation is based on: func:`scipy.ndimage.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius by which regions should be eroded.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None,
        a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension (number of axes).
        If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    eroded : ndarray of bool
        The result of the morphological erosion taking values in
        ``[False, True]``.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """
    dist = ndi.distance_transform_edt(image, sampling=spacing)
    return np.greater(dist, radius, out=out)