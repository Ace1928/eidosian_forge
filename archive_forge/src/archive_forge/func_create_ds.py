from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
def create_ds(var_names: list[list[str]], dim: bool=False, coord: bool=False, drop_idx: list[int] | None=None) -> list[Dataset]:
    out_ds = []
    ds = Dataset()
    ds = ds.assign_coords({'x': np.arange(2)})
    ds = ds.assign_coords({'y': np.arange(3)})
    ds = ds.assign_coords({'z': np.arange(4)})
    for i, dsl in enumerate(var_names):
        vlist = dsl.copy()
        if drop_idx is not None:
            vlist.pop(drop_idx[i])
        foo_data = np.arange(48, dtype=float).reshape(2, 2, 3, 4)
        dsi = ds.copy()
        if coord:
            dsi = ds.assign({'time': (['time'], [i * 2, i * 2 + 1])})
        for k in vlist:
            dsi = dsi.assign({k: (['time', 'x', 'y', 'z'], foo_data.copy())})
        if not dim:
            dsi = dsi.isel(time=0)
        out_ds.append(dsi)
    return out_ds