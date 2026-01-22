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
def test_2d_coords_line_plot(self) -> None:
    lon, lat = np.meshgrid(np.linspace(-20, 20, 5), np.linspace(0, 30, 4))
    lon += lat / 10
    lat += lon / 10
    da = xr.DataArray(np.arange(20).reshape(4, 5), dims=['y', 'x'], coords={'lat': (('y', 'x'), lat), 'lon': (('y', 'x'), lon)})
    with figure_context():
        hdl = da.plot.line(x='lon', hue='x')
        assert len(hdl) == 5
    with figure_context():
        hdl = da.plot.line(x='lon', hue='y')
        assert len(hdl) == 4
    with pytest.raises(ValueError, match='For 2D inputs, hue must be a dimension'):
        da.plot.line(x='lon', hue='lat')