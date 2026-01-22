import numpy as np
from scipy import signal
def subdivide_polygon(coords, degree=2, preserve_ends=False):
    """Subdivision of polygonal curves using B-Splines.

    Note that the resulting curve is always within the convex hull of the
    original polygon. Circular polygons stay closed after subdivision.

    Parameters
    ----------
    coords : (K, 2) array
        Coordinate array.
    degree : {1, 2, 3, 4, 5, 6, 7}, optional
        Degree of B-Spline. Default is 2.
    preserve_ends : bool, optional
        Preserve first and last coordinate of non-circular polygon. Default is
        False.

    Returns
    -------
    coords : (L, 2) array
        Subdivided coordinate array.

    References
    ----------
    .. [1] http://mrl.nyu.edu/publications/subdiv-course2000/coursenotes00.pdf
    """
    if degree not in _SUBDIVISION_MASKS:
        raise ValueError('Invalid B-Spline degree. Only degree 1 - 7 is supported.')
    circular = np.all(coords[0, :] == coords[-1, :])
    method = 'valid'
    if circular:
        coords = coords[:-1, :]
        method = 'same'
    mask_even, mask_odd = _SUBDIVISION_MASKS[degree]
    mask_even = np.array(mask_even, float) / 2 ** degree
    mask_odd = np.array(mask_odd, float) / 2 ** degree
    even = signal.convolve2d(coords.T, np.atleast_2d(mask_even), mode=method, boundary='wrap')
    odd = signal.convolve2d(coords.T, np.atleast_2d(mask_odd), mode=method, boundary='wrap')
    out = np.zeros((even.shape[1] + odd.shape[1], 2))
    out[1::2] = even.T
    out[::2] = odd.T
    if circular:
        out = np.vstack([out, out[0, :]])
    if preserve_ends and (not circular):
        out = np.vstack([coords[0, :], out, coords[-1, :]])
    return out