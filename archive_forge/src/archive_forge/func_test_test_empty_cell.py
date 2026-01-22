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
def test_test_empty_cell(self) -> None:
    g = self.darray.isel(row=1).drop_vars('row').plot(col='col', hue='hue', col_wrap=2)
    bottomright = g.axs[-1, -1]
    assert not bottomright.has_data()
    assert not bottomright.get_visible()