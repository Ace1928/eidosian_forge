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
@pytest.mark.parametrize('dim', ('x', 'y'))
def test_labels_with_units_with_interval(self, dim) -> None:
    """Test line plot with intervals and a units attribute."""
    bins = [-1, 0, 1, 2]
    arr = self.darray.groupby_bins('dim_0', bins).mean(...)
    arr.dim_0_bins.attrs['units'] = 'm'
    mappable, = arr.plot(**{dim: 'dim_0_bins'})
    ax = mappable.figure.gca()
    actual = getattr(ax, f'get_{dim}label')()
    expected = 'dim_0_bins_center [m]'
    assert actual == expected