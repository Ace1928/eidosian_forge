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
@ngjit
def combine_cpu_3d(aggs, selector_aggs):
    ny, nx, ncat = aggs[0].shape
    for y in range(ny):
        for x in range(nx):
            for cat in range(ncat):
                value = selector_aggs[1][y, x, cat]
                if not invalid(value) and append(x, y, selector_aggs[0][:, :, cat], value) >= 0:
                    aggs[0][y, x, cat] = aggs[1][y, x, cat]