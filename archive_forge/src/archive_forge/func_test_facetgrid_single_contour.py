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
@requires_matplotlib
def test_facetgrid_single_contour() -> None:
    x, y = np.meshgrid(np.arange(12), np.arange(12))
    z = xr.DataArray(np.sqrt(x ** 2 + y ** 2))
    z2 = xr.DataArray(np.sqrt(x ** 2 + y ** 2) + 1)
    ds = xr.concat([z, z2], dim='time')
    ds['time'] = [0, 1]
    with figure_context():
        ds.plot.contour(col='time', levels=[4], colors=['k'])