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
@ngjit_parallel
def nanmin_n_in_place_4d(ret, other):
    """Combine two min-n arrays, taking nans into account. Min-n arrays are 4D
    with shape (ny, nx, ncat, n) where ny and nx are the number of pixels,
    ncat the number of categories (will be 1 if not using a categorical
    reduction) and the last axis containing n values in ascending order.
    If there are fewer than n values it is padded with nans.
    Return the first array.
    """
    ny, nx, ncat, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            for cat in range(ncat):
                _nanmin_n_impl(ret[y, x, cat], other[y, x, cat])