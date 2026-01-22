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
def test_xyincrease_defaults(self) -> None:
    self.plotfunc(DataArray(easy_array((3, 2)), coords=[[1, 2, 3], [1, 2]]))
    bounds = plt.gca().get_ylim()
    assert bounds[0] < bounds[1]
    bounds = plt.gca().get_xlim()
    assert bounds[0] < bounds[1]
    self.plotfunc(DataArray(easy_array((3, 2)), coords=[[3, 2, 1], [2, 1]]))
    bounds = plt.gca().get_ylim()
    assert bounds[0] < bounds[1]
    bounds = plt.gca().get_xlim()
    assert bounds[0] < bounds[1]