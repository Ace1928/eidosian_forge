import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import _hough_circle, _hough_ellipse, _hough_line
from ._hough_transform import _probabilistic_hough_line as _prob_hough_line
def hough_ellipse(image, threshold=4, accuracy=1, min_size=4, max_size=None):
    """Perform an elliptical Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int, optional
        Accumulator threshold value. A lower value will return more ellipses.
    accuracy : double, optional
        Bin size on the minor axis used in the accumulator. A higher value
        will return more ellipses, but lead to a less precise estimation of
        the minor axis lengths.
    min_size : int, optional
        Minimal major axis length.
    max_size : int, optional
        Maximal minor axis length.
        If None, the value is set to half of the smaller
        image dimension.

    Returns
    -------
    result : ndarray with fields [(accumulator, yc, xc, a, b, orientation)].
        Where ``(yc, xc)`` is the center, ``(a, b)`` the major and minor
        axes, respectively. The `orientation` value follows the
        `skimage.draw.ellipse_perimeter` convention.

    Examples
    --------
    >>> from skimage.transform import hough_ellipse
    >>> from skimage.draw import ellipse_perimeter
    >>> img = np.zeros((25, 25), dtype=np.uint8)
    >>> rr, cc = ellipse_perimeter(10, 10, 6, 8)
    >>> img[cc, rr] = 1
    >>> result = hough_ellipse(img, threshold=8)
    >>> result.tolist()
    [(10, 10.0, 10.0, 8.0, 6.0, 0.0)]

    Notes
    -----
    Potential ellipses in the image are characterized by their major and
    minor axis lengths. For any pair of nonzero pixels in the image that
    are at least half of `min_size` apart, an accumulator keeps track of
    the minor axis lengths of potential ellipses formed with all the
    other nonzero pixels. If any bin (with `bin_size = accuracy * accuracy`)
    in the histogram of those accumulated minor axis lengths is above
    `threshold`, the corresponding ellipse is added to the results.

    A higher `accuracy` will therefore lead to more ellipses being found
    in the image, at the cost of a less precise estimation of the minor
    axis length.

    References
    ----------
    .. [1] Xie, Yonghong, and Qiang Ji. "A new efficient ellipse detection
           method." Pattern Recognition, 2002. Proceedings. 16th International
           Conference on. Vol. 2. IEEE, 2002
    """
    return _hough_ellipse(image, threshold=threshold, accuracy=accuracy, min_size=min_size, max_size=max_size)