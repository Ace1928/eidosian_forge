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
def test_contourf_cmap_set(self) -> None:
    a = DataArray(easy_array((4, 4)), dims=['z', 'time'])
    cmap_expected = mpl.colormaps['viridis']
    pl = a.plot.contourf(cmap=copy(cmap_expected), vmin=0.1, vmax=0.9)
    cmap = pl.cmap
    assert cmap is not None
    assert_array_equal(cmap(np.ma.masked_invalid([np.nan]))[0], cmap_expected(np.ma.masked_invalid([np.nan]))[0])
    assert cmap(-np.inf) == cmap_expected(-np.inf)
    assert cmap(np.inf) == cmap_expected(np.inf)