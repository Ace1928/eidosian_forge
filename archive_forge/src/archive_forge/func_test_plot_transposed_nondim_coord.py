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
@pytest.mark.parametrize('plotfunc', ['pcolormesh', 'contourf', 'contour'])
def test_plot_transposed_nondim_coord(plotfunc) -> None:
    x = np.linspace(0, 10, 101)
    h = np.linspace(3, 7, 101)
    s = np.linspace(0, 1, 51)
    z = s[:, np.newaxis] * h[np.newaxis, :]
    da = xr.DataArray(np.sin(x) * np.cos(z), dims=['s', 'x'], coords={'x': x, 's': s, 'z': (('s', 'x'), z), 'zt': (('x', 's'), z.T)})
    with figure_context():
        getattr(da.plot, plotfunc)(x='x', y='zt')
    with figure_context():
        getattr(da.plot, plotfunc)(x='zt', y='x')