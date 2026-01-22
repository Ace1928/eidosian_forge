import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import _hough_circle, _hough_ellipse, _hough_line
from ._hough_transform import _probabilistic_hough_line as _prob_hough_line
def hough_circle(image, radius, normalize=True, full_output=False):
    """Perform a circular Hough transform.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input image with nonzero values representing edges.
    radius : scalar or sequence of scalars
        Radii at which to compute the Hough transform.
        Floats are converted to integers.
    normalize : boolean, optional
        Normalize the accumulator with the number
        of pixels used to draw the radius.
    full_output : boolean, optional
        Extend the output size by twice the largest
        radius in order to detect centers outside the
        input picture.

    Returns
    -------
    H : ndarray, shape (radius index, M + 2R, N + 2R)
        Hough transform accumulator for each radius.
        R designates the larger radius if full_output is True.
        Otherwise, R = 0.

    Examples
    --------
    >>> from skimage.transform import hough_circle
    >>> from skimage.draw import circle_perimeter
    >>> img = np.zeros((100, 100), dtype=bool)
    >>> rr, cc = circle_perimeter(25, 35, 23)
    >>> img[rr, cc] = 1
    >>> try_radii = np.arange(5, 50)
    >>> res = hough_circle(img, try_radii)
    >>> ridx, r, c = np.unravel_index(np.argmax(res), res.shape)
    >>> r, c, try_radii[ridx]
    (25, 35, 23)

    """
    radius = np.atleast_1d(np.asarray(radius))
    return _hough_circle(image, radius.astype(np.intp), normalize=normalize, full_output=full_output)