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
def nanmin_n_in_place_3d(ret, other):
    """3d version of nanmin_n_in_place_4d, taking arrays of shape (ny, nx, n).
    """
    ny, nx, _n = ret.shape
    for y in nb.prange(ny):
        for x in range(nx):
            _nanmin_n_impl(ret[y, x], other[y, x])