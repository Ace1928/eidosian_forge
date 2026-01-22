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
def to_cupy_array(df, columns):
    if isinstance(columns, tuple):
        columns = list(columns)
    if isinstance(columns, list) and len(columns) != len(set(columns)):
        return cp.stack([cp.array(df[c]) for c in columns], axis=1)
    if Version(cudf.__version__) >= Version('22.02'):
        return df[columns].to_cupy()
    else:
        if not isinstance(columns, list):
            return df[columns].to_gpu_array()
        return df[columns].as_gpu_matrix()