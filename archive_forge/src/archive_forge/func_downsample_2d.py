from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
def downsample_2d(src, w, h, method=DS_MEAN, fill_value=None, mode_rank=1, out=None):
    """
    Downsample a 2-D grid to a lower resolution by aggregating original grid cells.

    Parameters
    ----------
    src : numpy.ndarray or dask.array.Array
        The source array to resample
    w : int
        New grid width
    h : int
        New grid height
    ds_method : str (optional)
        Grid cell aggregation method for a possible downsampling
        (one of the *DS_* constants).
    fill_value : scalar (optional)
        If ``None``, it is taken from **src** if it is a masked array,
        otherwise from *out* if it is a masked array,
        otherwise numpy's default value is used.
    mode_rank : scalar (optional)
        The rank of the frequency determined by the *ds_method*
        ``DS_MODE``. One (the default) means most frequent value, two
        means second most frequent value, and so forth.
    out : numpy.ndarray (optional)
        Alternate output array in which to place the result. The
        default is *None*; if provided, it must have the same shape as
        the expected output.

    Returns
    -------
    downsampled : numpy.ndarray or dask.array.Array
        An downsampled version of the *src* array.
    """
    if method == DS_MODE and mode_rank < 1:
        raise ValueError('mode_rank must be >= 1')
    out = _get_out(out, src, (h, w))
    if out is None:
        return src
    mask, use_mask = _get_mask(src)
    fill_value = _get_fill_value(fill_value, src, out)
    if method not in DOWNSAMPLING_METHODS:
        raise ValueError('invalid downsampling method')
    downsampling_method = DOWNSAMPLING_METHODS[method]
    downsampled = downsampling_method(src, mask, use_mask, method, fill_value, mode_rank, (0, 0), (0, 0), out)
    return _mask_or_not(downsampled, src, fill_value)