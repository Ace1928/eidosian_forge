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
@ngjit
def row_min_n_in_place_3d(ret, other):
    ny, nx, _n = ret.shape
    for y in range(ny):
        for x in range(nx):
            _row_min_n_impl(ret[y, x], other[y, x])