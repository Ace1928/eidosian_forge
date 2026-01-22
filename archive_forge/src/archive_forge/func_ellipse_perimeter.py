import numpy as np
from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require
from .._shared.compat import NP_COPY_IF_NEEDED
from ._draw import (
def ellipse_perimeter(r, c, r_radius, c_radius, orientation=0, shape=None):
    """Generate ellipse perimeter coordinates.

    Parameters
    ----------
    r, c : int
        Centre coordinate of ellipse.
    r_radius, c_radius : int
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    orientation : double, optional
        Major axis orientation in clockwise direction as radians.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for ellipses that exceed the image
        size. If None, the full extent of the ellipse is used.  Must be at
        least length 2. Only the first two values are used to determine the
        extent of the input image.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the ellipse perimeter.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf

    Examples
    --------
    >>> from skimage.draw import ellipse_perimeter
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = ellipse_perimeter(5, 5, 3, 4)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)


    Note that the positions of `ellipse` without specified `shape` can have
    also, negative values, as this is correct on the plane. On the other hand
    using these ellipse positions for an image afterwards may lead to appearing
    on the other side of image, because ``image[-1, -1] = image[end-1, end-1]``

    >>> rr, cc = ellipse_perimeter(2, 3, 4, 5)
    >>> img = np.zeros((9, 12), dtype=np.uint8)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return _ellipse_perimeter(r, c, r_radius, c_radius, orientation, shape)