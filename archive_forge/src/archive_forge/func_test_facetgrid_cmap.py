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
def test_facetgrid_cmap(self) -> None:
    data = np.random.random(size=(20, 25, 12)) + np.linspace(-3, 3, 12)
    d = DataArray(data, dims=['x', 'y', 'time'])
    fg = d.plot.pcolormesh(col='time')
    assert len({m.get_clim() for m in fg._mappables}) == 1
    assert len({m.get_cmap().name for m in fg._mappables}) == 1