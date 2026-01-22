import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
@staticmethod
def validate_input(x, y, w, bbox, k, s, ext, check_finite):
    x, y, bbox = (np.asarray(x), np.asarray(y), np.asarray(bbox))
    if w is not None:
        w = np.asarray(w)
    if check_finite:
        w_finite = np.isfinite(w).all() if w is not None else True
        if not np.isfinite(x).all() or not np.isfinite(y).all() or (not w_finite):
            raise ValueError('x and y array must not contain NaNs or infs.')
    if s is None or s > 0:
        if not np.all(diff(x) >= 0.0):
            raise ValueError('x must be increasing if s > 0')
    elif not np.all(diff(x) > 0.0):
        raise ValueError('x must be strictly increasing if s = 0')
    if x.size != y.size:
        raise ValueError('x and y should have a same length')
    elif w is not None and (not x.size == y.size == w.size):
        raise ValueError('x, y, and w should have a same length')
    elif bbox.shape != (2,):
        raise ValueError('bbox shape should be (2,)')
    elif not 1 <= k <= 5:
        raise ValueError('k should be 1 <= k <= 5')
    elif s is not None and (not s >= 0.0):
        raise ValueError('s should be s >= 0.0')
    try:
        ext = _extrap_modes[ext]
    except KeyError as e:
        raise ValueError('Unknown extrapolation mode %s.' % ext) from e
    return (x, y, w, bbox, ext)