import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import _hough_circle, _hough_ellipse, _hough_line
from ._hough_transform import _probabilistic_hough_line as _prob_hough_line
def hough_circle_peaks(hspaces, radii, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=np.inf, total_num_peaks=np.inf, normalize=False):
    """Return peaks in a circle Hough transform.

    Identifies most prominent circles separated by certain distances in given
    Hough spaces. Non-maximum suppression with different sizes is applied
    separately in the first and second dimension of the Hough space to
    identify peaks. For circles with different radius but close in distance,
    only the one with highest peak is kept.

    Parameters
    ----------
    hspaces : (M, N, P) array
        Hough spaces returned by the `hough_circle` function.
    radii : (M,) array
        Radii corresponding to Hough spaces.
    min_xdistance : int, optional
        Minimum distance separating centers in the x dimension.
    min_ydistance : int, optional
        Minimum distance separating centers in the y dimension.
    threshold : float, optional
        Minimum intensity of peaks in each Hough space.
        Default is `0.5 * max(hspace)`.
    num_peaks : int, optional
        Maximum number of peaks in each Hough space. When the
        number of peaks exceeds `num_peaks`, only `num_peaks`
        coordinates based on peak intensity are considered for the
        corresponding radius.
    total_num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.
    normalize : bool, optional
        If True, normalize the accumulator by the radius to sort the prominent
        peaks.

    Returns
    -------
    accum, cx, cy, rad : tuple of array
        Peak values in Hough space, x and y center coordinates and radii.

    Examples
    --------
    >>> from skimage import transform, draw
    >>> img = np.zeros((120, 100), dtype=int)
    >>> radius, x_0, y_0 = (20, 99, 50)
    >>> y, x = draw.circle_perimeter(y_0, x_0, radius)
    >>> img[x, y] = 1
    >>> hspaces = transform.hough_circle(img, radius)
    >>> accum, cx, cy, rad = hough_circle_peaks(hspaces, [radius,])

    Notes
    -----
    Circles with bigger radius have higher peaks in Hough space. If larger
    circles are preferred over smaller ones, `normalize` should be False.
    Otherwise, circles will be returned in the order of decreasing voting
    number.
    """
    from ..feature.peak import _prominent_peaks
    r = []
    cx = []
    cy = []
    accum = []
    for rad, hp in zip(radii, hspaces):
        h_p, x_p, y_p = _prominent_peaks(hp, min_xdistance=min_xdistance, min_ydistance=min_ydistance, threshold=threshold, num_peaks=num_peaks)
        r.extend((rad,) * len(h_p))
        cx.extend(x_p)
        cy.extend(y_p)
        accum.extend(h_p)
    r = np.array(r)
    cx = np.array(cx)
    cy = np.array(cy)
    accum = np.array(accum)
    if normalize:
        s = np.argsort(accum / r)
    else:
        s = np.argsort(accum)
    accum_sorted, cx_sorted, cy_sorted, r_sorted = (accum[s][::-1], cx[s][::-1], cy[s][::-1], r[s][::-1])
    tnp = len(accum_sorted) if total_num_peaks == np.inf else total_num_peaks
    if min_xdistance == 1 and min_ydistance == 1 or len(accum_sorted) == 0:
        return (accum_sorted[:tnp], cx_sorted[:tnp], cy_sorted[:tnp], r_sorted[:tnp])
    should_keep = label_distant_points(cx_sorted, cy_sorted, min_xdistance, min_ydistance, tnp)
    return (accum_sorted[should_keep], cx_sorted[should_keep], cy_sorted[should_keep], r_sorted[should_keep])