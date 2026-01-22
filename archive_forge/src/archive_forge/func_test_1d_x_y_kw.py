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
def test_1d_x_y_kw(self) -> None:
    z = np.arange(10)
    da = DataArray(np.cos(z), dims=['z'], coords=[z], name='f')
    xy: list[list[None | str]] = [[None, None], [None, 'z'], ['z', None]]
    f, ax = plt.subplots(3, 1)
    for aa, (x, y) in enumerate(xy):
        da.plot(x=x, y=y, ax=ax.flat[aa])
    with pytest.raises(ValueError, match='Cannot specify both'):
        da.plot(x='z', y='z')
    error_msg = "must be one of None, 'z'"
    with pytest.raises(ValueError, match=f'x {error_msg}'):
        da.plot(x='f')
    with pytest.raises(ValueError, match=f'y {error_msg}'):
        da.plot(y='f')