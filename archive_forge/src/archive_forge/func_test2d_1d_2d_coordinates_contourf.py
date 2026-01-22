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
def test2d_1d_2d_coordinates_contourf(self) -> None:
    sz = (20, 10)
    depth = easy_array(sz)
    a = DataArray(easy_array(sz), dims=['z', 'time'], coords={'depth': (['z', 'time'], depth), 'time': np.linspace(0, 1, sz[1])})
    a.plot.contourf(x='time', y='depth')
    a.plot.contourf(x='depth', y='time')