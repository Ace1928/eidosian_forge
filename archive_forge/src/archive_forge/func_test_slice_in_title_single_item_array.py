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
def test_slice_in_title_single_item_array(self) -> None:
    """Edge case for data of shape (1, N) or (N, 1)."""
    darray = self.darray.expand_dims({'d': np.array([10.009])})
    darray.plot.line(x='period')
    title = plt.gca().get_title()
    assert 'd = 10.01' == title