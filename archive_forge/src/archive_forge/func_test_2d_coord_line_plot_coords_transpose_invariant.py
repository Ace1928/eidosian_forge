from __future__ import annotations
import contextlib
import inspect
import math
from collections.abc import Hashable
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.plot as xplt
from xarray import DataArray, Dataset
from xarray.namedarray.utils import module_available
from xarray.plot.dataarray_plot import _infer_interval_breaks
from xarray.plot.dataset_plot import _infer_meta_data
from xarray.plot.utils import (
from xarray.tests import (
def test_2d_coord_line_plot_coords_transpose_invariant(self) -> None:
    x = np.arange(10)
    y = np.arange(20)
    ds = xr.Dataset(coords={'x': x, 'y': y})
    for z in [ds.y + ds.x, ds.x + ds.y]:
        ds = ds.assign_coords(z=z)
        ds['v'] = ds.x + ds.y
        ds['v'].plot.line(y='z', hue='x')