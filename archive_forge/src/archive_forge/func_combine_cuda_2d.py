from __future__ import annotations
import copy
from enum import Enum
from packaging.version import Version
import numpy as np
from datashader.datashape import dshape, isnumeric, Record, Option
from datashader.datashape import coretypes as ct
from toolz import concat, unique
import xarray as xr
from datashader.antialias import AntialiasCombination, AntialiasStage2
from datashader.utils import isminus1, isnull
from numba import cuda as nb_cuda
from .utils import (
@nb_cuda.jit
def combine_cuda_2d(aggs, selector_aggs):
    ny, nx = aggs[0].shape
    x, y = nb_cuda.grid(2)
    if x < nx and y < ny:
        value = selector_aggs[1][y, x]
        if not invalid(value) and append(x, y, selector_aggs[0], value) >= 0:
            aggs[0][y, x] = aggs[1][y, x]