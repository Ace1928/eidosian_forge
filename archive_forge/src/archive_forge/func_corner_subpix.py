import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD
def corner_subpix(image, corners, window_size=11, alpha=0.99):
    """Determine subpixel position of corners.

    A statistical test decides whether the corner is defined as the
    intersection of two edges or a single peak. Depending on the classification
    result, the subpixel corner location is determined based on the local
    covariance of the grey-values. If the significance level for either
    statistical test is not sufficient, the corner cannot be classified, and
    the output subpixel position is set to NaN.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    corners : (K, 2) ndarray
        Corner coordinates `(row, col)`.
    window_size : int, optional
        Search window size for subpixel estimation.
    alpha : float, optional
        Significance level for corner classification.

    Returns
    -------
    positions : (K, 2) ndarray
        Subpixel corner positions. NaN for "not classified" corners.

    References
    ----------
    .. [1] Förstner, W., & Gülch, E. (1987, June). A fast operator for
           detection and precise location of distinct points, corners and
           centres of circular features. In Proc. ISPRS intercommission
           conference on fast processing of photogrammetric data (pp. 281-305).
           https://cseweb.ucsd.edu/classes/sp02/cse252/foerstner/foerstner.pdf
    .. [2] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_harris, corner_peaks, corner_subpix
    >>> img = np.zeros((10, 10))
    >>> img[:5, :5] = 1
    >>> img[5:, 5:] = 1
    >>> img.astype(int)
    array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> coords = corner_peaks(corner_harris(img), min_distance=2)
    >>> coords_subpix = corner_subpix(img, coords, window_size=7)
    >>> coords_subpix
    array([[4.5, 4.5]])

    """
    wext = (window_size - 1) // 2
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    image = np.pad(image, pad_width=wext, mode='constant', constant_values=0)
    corners = safe_as_int(corners + wext)
    N_dot = np.zeros((2, 2), dtype=float_dtype)
    N_edge = np.zeros((2, 2), dtype=float_dtype)
    b_dot = np.zeros((2,), dtype=float_dtype)
    b_edge = np.zeros((2,), dtype=float_dtype)
    redundancy = window_size ** 2 - 2
    t_crit_dot = stats.f.isf(1 - alpha, redundancy, redundancy)
    t_crit_edge = stats.f.isf(alpha, redundancy, redundancy)
    y, x = np.mgrid[-wext:wext + 1, -wext:wext + 1]
    corners_subpix = np.zeros_like(corners, dtype=float_dtype)
    for i, (y0, x0) in enumerate(corners):
        miny = y0 - wext - 1
        maxy = y0 + wext + 2
        minx = x0 - wext - 1
        maxx = x0 + wext + 2
        window = image[miny:maxy, minx:maxx]
        winy, winx = _compute_derivatives(window, mode='constant', cval=0)
        winx_winx = (winx * winx)[1:-1, 1:-1]
        winx_winy = (winx * winy)[1:-1, 1:-1]
        winy_winy = (winy * winy)[1:-1, 1:-1]
        Axx = np.sum(winx_winx)
        Axy = np.sum(winx_winy)
        Ayy = np.sum(winy_winy)
        bxx_x = np.sum(winx_winx * x)
        bxx_y = np.sum(winx_winx * y)
        bxy_x = np.sum(winx_winy * x)
        bxy_y = np.sum(winx_winy * y)
        byy_x = np.sum(winy_winy * x)
        byy_y = np.sum(winy_winy * y)
        N_dot[0, 0] = Axx
        N_dot[0, 1] = N_dot[1, 0] = -Axy
        N_dot[1, 1] = Ayy
        N_edge[0, 0] = Ayy
        N_edge[0, 1] = N_edge[1, 0] = Axy
        N_edge[1, 1] = Axx
        b_dot[:] = (bxx_y - bxy_x, byy_x - bxy_y)
        b_edge[:] = (byy_y + bxy_x, bxx_x + bxy_y)
        try:
            est_dot = np.linalg.solve(N_dot, b_dot)
            est_edge = np.linalg.solve(N_edge, b_edge)
        except np.linalg.LinAlgError:
            corners_subpix[i, :] = (np.nan, np.nan)
            continue
        ry_dot = y - est_dot[0]
        rx_dot = x - est_dot[1]
        ry_edge = y - est_edge[0]
        rx_edge = x - est_edge[1]
        rxx_dot = rx_dot * rx_dot
        rxy_dot = rx_dot * ry_dot
        ryy_dot = ry_dot * ry_dot
        rxx_edge = rx_edge * rx_edge
        rxy_edge = rx_edge * ry_edge
        ryy_edge = ry_edge * ry_edge
        var_dot = np.sum(winx_winx * ryy_dot - 2 * winx_winy * rxy_dot + winy_winy * rxx_dot)
        var_edge = np.sum(winy_winy * ryy_edge + 2 * winx_winy * rxy_edge + winx_winx * rxx_edge)
        if var_dot < np.spacing(1) and var_edge < np.spacing(1):
            t = np.nan
        elif var_dot == 0:
            t = np.inf
        else:
            t = var_edge / var_dot
        corner_class = int(t < t_crit_edge) - int(t > t_crit_dot)
        if corner_class == -1:
            corners_subpix[i, :] = (y0 + est_dot[0], x0 + est_dot[1])
        elif corner_class == 0:
            corners_subpix[i, :] = (np.nan, np.nan)
        elif corner_class == 1:
            corners_subpix[i, :] = (y0 + est_edge[0], x0 + est_edge[1])
    corners_subpix -= wext
    return corners_subpix