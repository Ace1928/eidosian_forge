from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
def upsample_2d(src, w, h, method=US_LINEAR, fill_value=None, out=None):
    """
    Upsample a 2-D grid to a higher resolution by interpolating original grid cells.

    src: 2-D *ndarray*
    w: *int*
        Grid width, which must be greater than or equal to *src.shape[-1]*
    h:  *int*
        Grid height, which must be greater than or equal to *src.shape[-2]*
    method: one of the *US_* constants, optional
        Grid cell interpolation method
    fill_value: *scalar*, optional
        If ``None``, it is taken from **src** if it is a masked array,
        otherwise from *out* if it is a masked array,
        otherwise numpy's default value is used.
    out: 2-D *ndarray*, optional
        Alternate output array in which to place the result. The default is *None*; if provided,
        it must have the same shape as the expected output.

    Returns
    -------
    upsampled : numpy.ndarray or dask.array.Array
        An upsampled version of the *src* array.
    """
    out = _get_out(out, src, (h, w))
    if out is None:
        return src
    mask, use_mask = _get_mask(src)
    fill_value = _get_fill_value(fill_value, src, out)
    if method not in UPSAMPLING_METHODS:
        raise ValueError('invalid upsampling method')
    upsampling_method = UPSAMPLING_METHODS[method]
    upsampled = upsampling_method(src, mask, use_mask, fill_value, (0, 0), (0, 0), out)
    return _mask_or_not(upsampled, src, fill_value)