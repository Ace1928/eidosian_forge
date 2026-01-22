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
def combine_cuda_n_4d(aggs, selector_aggs):
    ny, nx, ncat, n = aggs[0].shape
    x, y, cat = nb_cuda.grid(3)
    if x < nx and y < ny and (cat < ncat):
        for i in range(n):
            value = selector_aggs[1][y, x, cat, i]
            if invalid(value):
                break
            update_index = append(x, y, selector_aggs[0][:, :, cat, :], value)
            if update_index < 0:
                break
            cuda_shift_and_insert(aggs[0][y, x, cat], aggs[1][y, x, cat, i], update_index)