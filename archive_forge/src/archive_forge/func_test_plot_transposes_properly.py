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
@pytest.mark.parametrize('plotfunc', ['pcolormesh', 'imshow'])
def test_plot_transposes_properly(plotfunc) -> None:
    da = xr.DataArray([np.sin(2 * np.pi / 10 * np.arange(10))] * 10, dims=('y', 'x'))
    with figure_context():
        hdl = getattr(da.plot, plotfunc)(x='x', y='y')
        assert_array_equal(hdl.get_array().ravel(), da.to_masked_array().ravel())