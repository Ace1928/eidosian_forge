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
def test_datetime_dimension(self) -> None:
    nrow = 3
    ncol = 4
    time = pd.date_range('2000-01-01', periods=nrow)
    a = DataArray(easy_array((nrow, ncol)), coords=[('time', time), ('y', range(ncol))])
    a.plot()
    ax = plt.gca()
    assert ax.has_data()