import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def to_rgba_array(c, alpha=None):
    """
    Convert *c* to a (n, 4) array of RGBA colors.

    Parameters
    ----------
    c : Matplotlib color or array of colors
        If *c* is a masked array, an `~numpy.ndarray` is returned with a
        (0, 0, 0, 0) row for each masked value or row in *c*.

    alpha : float or sequence of floats, optional
        If *alpha* is given, force the alpha value of the returned RGBA tuple
        to *alpha*.

        If None, the alpha value from *c* is used. If *c* does not have an
        alpha channel, then alpha defaults to 1.

        *alpha* is ignored for the color value ``"none"`` (case-insensitive),
        which always maps to ``(0, 0, 0, 0)``.

        If *alpha* is a sequence and *c* is a single color, *c* will be
        repeated to match the length of *alpha*.

    Returns
    -------
    array
        (n, 4) array of RGBA colors,  where each channel (red, green, blue,
        alpha) can assume values between 0 and 1.
    """
    if isinstance(c, tuple) and len(c) == 2 and isinstance(c[1], Real):
        if alpha is None:
            c, alpha = c
        else:
            c = c[0]
    if np.iterable(alpha):
        alpha = np.asarray(alpha).ravel()
    if isinstance(c, np.ndarray) and c.dtype.kind in 'if' and (c.ndim == 2) and (c.shape[1] in [3, 4]):
        mask = c.mask.any(axis=1) if np.ma.is_masked(c) else None
        c = np.ma.getdata(c)
        if np.iterable(alpha):
            if c.shape[0] == 1 and alpha.shape[0] > 1:
                c = np.tile(c, (alpha.shape[0], 1))
            elif c.shape[0] != alpha.shape[0]:
                raise ValueError('The number of colors must match the number of alpha values if there are more than one of each.')
        if c.shape[1] == 3:
            result = np.column_stack([c, np.zeros(len(c))])
            result[:, -1] = alpha if alpha is not None else 1.0
        elif c.shape[1] == 4:
            result = c.copy()
            if alpha is not None:
                result[:, -1] = alpha
        if mask is not None:
            result[mask] = 0
        if np.any((result < 0) | (result > 1)):
            raise ValueError('RGBA values should be within 0-1 range')
        return result
    if cbook._str_lower_equal(c, 'none'):
        return np.zeros((0, 4), float)
    try:
        if np.iterable(alpha):
            return np.array([to_rgba(c, a) for a in alpha], float)
        else:
            return np.array([to_rgba(c, alpha)], float)
    except TypeError:
        pass
    except ValueError as e:
        if e.args == ("'alpha' must be between 0 and 1, inclusive",):
            raise e
    if isinstance(c, str):
        raise ValueError(f'{c!r} is not a valid color value.')
    if len(c) == 0:
        return np.zeros((0, 4), float)
    if isinstance(c, Sequence):
        lens = {len(cc) if isinstance(cc, (list, tuple)) else -1 for cc in c}
        if lens == {3}:
            rgba = np.column_stack([c, np.ones(len(c))])
        elif lens == {4}:
            rgba = np.array(c)
        else:
            rgba = np.array([to_rgba(cc) for cc in c])
    else:
        rgba = np.array([to_rgba(cc) for cc in c])
    if alpha is not None:
        rgba[:, 3] = alpha
    return rgba