from __future__ import annotations
from packaging.version import Version
import inspect
import warnings
import os
from math import isnan
import numpy as np
import pandas as pd
import xarray as xr
from datashader.utils import Expr, ngjit
from datashader.macros import expand_varargs
@staticmethod
def maybe_expand_bounds(bounds):
    minval, maxval = bounds
    if not (np.isfinite(minval) and np.isfinite(maxval)):
        minval, maxval = (-1.0, 1.0)
    elif minval == maxval:
        minval, maxval = (minval - 1, minval + 1)
    return (minval, maxval)