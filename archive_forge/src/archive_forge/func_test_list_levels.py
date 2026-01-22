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
def test_list_levels(self) -> None:
    data = self.data + 1
    orig_levels = [0, 1, 2, 3, 4, 5]
    cmap_params = _determine_cmap_params(data, levels=orig_levels, vmin=0, vmax=3)
    assert cmap_params['vmin'] is None
    assert cmap_params['vmax'] is None
    assert cmap_params['norm'].vmin == 0
    assert cmap_params['norm'].vmax == 5
    assert cmap_params['cmap'].N == 5
    assert cmap_params['norm'].N == 6
    for wrap_levels in [list, np.array, pd.Index, DataArray]:
        cmap_params = _determine_cmap_params(data, levels=wrap_levels(orig_levels))
        assert_array_equal(cmap_params['levels'], orig_levels)