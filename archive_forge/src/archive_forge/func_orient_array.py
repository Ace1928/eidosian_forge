from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
def orient_array(raster, res=None, layer=None):
    """
    Reorients the array to a canonical orientation depending on
    whether the x and y-resolution values are positive or negative.

    Parameters
    ----------
    raster : DataArray
        xarray DataArray to be reoriented
    res : tuple
        Two-tuple (int, int) which includes x and y resolutions (aka "grid/cell
        sizes"), respectively.
    layer : int
        Index of the raster layer to be reoriented (optional)

    Returns
    -------
    array : numpy.ndarray
        Reoriented 2d NumPy ndarray
    """
    if res is None:
        res = calc_res(raster)
    array = raster.data
    if layer is not None:
        array = array[layer - 1]
    r0zero = np.timedelta64(0, 'ns') if isinstance(res[0], np.timedelta64) else 0
    r1zero = np.timedelta64(0, 'ns') if isinstance(res[1], np.timedelta64) else 0
    xflip = res[0] < r0zero
    yflip = res[1] > r1zero
    array = _flip_array(array, xflip, yflip)
    return array